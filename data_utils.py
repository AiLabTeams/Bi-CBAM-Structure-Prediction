import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

try:
    import openpyxl
except ImportError:
    openpyxl = None


SOURCE_IMAGE_SIZE = (138, 80)
IMAGE_CHANNELS = 1


DATA_TRANSFORM = transforms.Compose([
    transforms.Resize(SOURCE_IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
    transforms.ToTensor(),
])


def canonical_imagename(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return "{:g}".format(f)
    except ValueError:
        if s.endswith(".0"):
            return s[:-2]
        return s


def read_labels_from_excel(excel_path: Path):
    try:
        if str(excel_path).lower().endswith(("xlsx", "xlsm", "xls")):
            if openpyxl is not None:
                df = pd.read_excel(excel_path, engine="openpyxl")
            else:
                df = pd.read_excel(excel_path)
        else:
            df = pd.read_excel(excel_path)
    except ImportError as exc:
        raise ImportError("Reading Excel requires openpyxl. Please install: pip install openpyxl") from exc

    df.columns = [str(c).strip() for c in df.columns]
    required = ["ImageName", "Times", "IA", "SphereRad", "OADphi1", "OADphi2", "OADphi3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")
    df = df.dropna(how="all")
    df["ImageStem"] = df["ImageName"].apply(canonical_imagename)
    return df


def encode_angle_deg(phi_deg: float):
    rad = math.radians(float(phi_deg) % 360.0)
    return math.sin(rad), math.cos(rad)


def build_samples(data_root: Path, excel_path: Path):
    df = read_labels_from_excel(excel_path)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_map = {}
    for p in data_root.iterdir():
        if p.is_file() and p.suffix.lower() in valid_exts:
            image_map[p.stem.lower()] = p

    samples = []

    def get_float(v, default):
        try:
            return float(v)
        except Exception:
            return default

    for _, row in df.iterrows():
        stem = row["ImageStem"]
        if not stem:
            continue

        stem_s = str(stem).lower()
        img_path = image_map.get(stem_s)
        if img_path is None:
            for width in range(2, 7):
                padded = stem_s.zfill(width)
                if padded in image_map:
                    img_path = image_map[padded]
                    break
        if img_path is None:
            continue

        times = int(np.clip(round(get_float(row["Times"], 1.0)), 1, 3))
        ia = float(np.clip(get_float(row["IA"], 0.0), 0.0, 90.0))
        rad = float(np.clip(get_float(row["SphereRad"], 50.0), 50.0, 250.0))
        phi1 = get_float(row["OADphi1"], 0.0) % 360.0
        phi2 = get_float(row["OADphi2"], 0.0) % 360.0
        phi3 = get_float(row["OADphi3"], 0.0) % 360.0
        if times < 2:
            phi2 = 0.0
        if times < 3:
            phi3 = 0.0

        sin1, cos1 = encode_angle_deg(phi1)
        sin2, cos2 = encode_angle_deg(phi2)
        sin3, cos3 = encode_angle_deg(phi3)

        samples.append({
            "img_path": str(img_path),
            "enc8": np.asarray([
                ia / 90.0,
                (rad - 50.0) / 200.0,
                sin1, cos1,
                sin2, cos2,
                sin3, cos3,
            ], dtype=np.float32),
            "times_idx": np.int64(times - 1),
            "physical_norm6": np.asarray([
                ia / 90.0,
                phi1 / 360.0,
                (rad - 50.0) / 200.0,
                times / 3.0,
                phi2 / 360.0,
                phi3 / 360.0,
            ], dtype=np.float32),
            "physical_raw6": np.asarray([ia, phi1, rad, float(times), phi2, phi3], dtype=np.float32),
        })

    if not samples:
        raise RuntimeError("No matched images found. Please check --data-root and --excel-path.")
    return samples


def image_complexity_score(img_path: str, transform=DATA_TRANSFORM):
    with Image.open(img_path) as img:
        img = img.convert("L")
        tensor = transform(img)
    x = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    gx = np.abs(np.diff(x, axis=1, prepend=x[:, :1]))
    gy = np.abs(np.diff(x, axis=0, prepend=x[:1, :]))
    edge_density = float((gx + gy).mean())
    fill_ratio = float((x > 0.5).mean())
    variance = float(x.var())
    return 0.5 * edge_density + 0.3 * fill_ratio + 0.2 * variance


def assign_difficulty_bins(samples_subset, transform=DATA_TRANSFORM):
    scores = np.asarray([image_complexity_score(s["img_path"], transform) for s in samples_subset], dtype=np.float32)
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    enriched = []
    for sample, score in zip(samples_subset, scores):
        if score <= q1:
            difficulty = "low"
        elif score <= q2:
            difficulty = "mid"
        else:
            difficulty = "high"
        enriched.append({**sample, "complexity_score": float(score), "difficulty": difficulty})
    return enriched


def stratified_select_test_subset(samples_subset, n_select, transform=DATA_TRANSFORM, seed=42):
    rng = np.random.RandomState(seed)
    enriched = assign_difficulty_bins(samples_subset, transform)
    eligible = [sample for sample in enriched if sample["difficulty"] in {"mid", "high"}]
    if n_select > len(eligible):
        raise ValueError(f"Requested {n_select} samples but only {len(eligible)} eligible test samples exist.")

    groups = {}
    for sample in eligible:
        key = (int(sample["times_idx"]) + 1, sample["difficulty"])
        groups.setdefault(key, []).append(sample)

    keys = sorted(groups.keys())
    total = len(eligible)
    base_counts = {k: int(np.floor(len(groups[k]) * n_select / total)) for k in keys}
    assigned = sum(base_counts.values())
    remainders = sorted(((len(groups[k]) * n_select / total - base_counts[k], k) for k in keys), reverse=True)
    for _, key in remainders[: n_select - assigned]:
        base_counts[key] += 1

    selected = []
    for key in keys:
        group = groups[key]
        take = min(base_counts[key], len(group))
        if take > 0:
            chosen_idx = rng.choice(len(group), size=take, replace=False)
            selected.extend([group[i] for i in chosen_idx])

    if len(selected) < n_select:
        remaining = [s for s in eligible if s["img_path"] not in {x["img_path"] for x in selected}]
        extra_idx = rng.choice(len(remaining), size=n_select - len(selected), replace=False)
        selected.extend([remaining[i] for i in extra_idx])

    rng.shuffle(selected)
    return selected[:n_select]


def split_indices(num_samples: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    index = np.arange(num_samples)
    rng.shuffle(index)
    n_train = int(num_samples * 0.7)
    n_val = int(num_samples * 0.2)
    train_idx = index[0:n_train].tolist()
    val_idx = index[n_train:n_train + n_val].tolist()
    test_idx = index[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


class LazyImageLabelDataset(Data.Dataset):
    def __init__(self, samples, transform=DATA_TRANSFORM):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with Image.open(s["img_path"]) as img:
            img = img.convert("L")
            x = self.transform(img)
        target = {
            "enc8": torch.tensor(s["enc8"], dtype=torch.float32),
            "times_idx": torch.tensor(s["times_idx"], dtype=torch.long),
            "physical_norm6": torch.tensor(s["physical_norm6"], dtype=torch.float32),
            "physical_raw6": torch.tensor(s["physical_raw6"], dtype=torch.float32),
        }
        return x, target
