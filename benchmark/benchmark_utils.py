import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

try:
    from sklearn.metrics import r2_score
except ImportError:
    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_outputs"
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "data" / "images"
DEFAULT_EXCEL_PATH = PROJECT_ROOT.parent / "data" / "labels.xlsx"
SEED = 42
MODEL_INPUT_SIZE = (138, 80)
IMAGE_CHANNELS = 1

DATA_TRANSFORM = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
    transforms.ToTensor(),
])


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def read_labels_from_excel(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, engine="openpyxl" if str(excel_path).lower().endswith("xlsx") else None)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["ImageName", "Times", "IA", "SphereRad", "OADphi1", "OADphi2", "OADphi3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")
    df = df.dropna(how="all").copy()
    df["ImageStem"] = df["ImageName"].apply(canonical_imagename)
    return df


def normalize_label(ia: float, phi1: float, rad: float, times: float, phi2: float, phi3: float) -> np.ndarray:
    phi1 = phi1 % 360.0
    phi2 = phi2 % 360.0
    phi3 = phi3 % 360.0
    if times < 2.0:
        phi2 = 0.0
    if times < 3.0:
        phi3 = 0.0
    return np.asarray([
        np.clip(ia, 0.0, 90.0) / 90.0,
        phi1 / 360.0,
        (np.clip(rad, 50.0, 250.0) - 50.0) / 200.0,
        np.clip(times, 1.0, 3.0) / 3.0,
        phi2 / 360.0,
        phi3 / 360.0,
    ], dtype=np.float32)


def encode_angle_deg(phi_deg: float) -> Tuple[float, float]:
    rad = math.radians(float(phi_deg) % 360.0)
    return math.sin(rad), math.cos(rad)


def pack_physical_norm6_np(norm6: Sequence[float]) -> np.ndarray:
    norm6 = np.asarray(norm6, dtype=np.float32)
    ia = float(np.clip(norm6[0], 0.0, 1.0))
    phi1 = float(norm6[1] % 1.0) * 360.0
    rad = float(np.clip(norm6[2], 0.0, 1.0))
    times_idx = int(np.clip(np.round(norm6[3] * 3.0), 1, 3)) - 1
    times = times_idx + 1
    phi2 = float(norm6[4] % 1.0) * 360.0 if times >= 2 else 0.0
    phi3 = float(norm6[5] % 1.0) * 360.0 if times >= 3 else 0.0
    sin1, cos1 = encode_angle_deg(phi1)
    sin2, cos2 = encode_angle_deg(phi2)
    sin3, cos3 = encode_angle_deg(phi3)
    times_onehot = np.zeros(3, dtype=np.float32)
    times_onehot[times_idx] = 1.0
    return np.asarray([ia, rad, sin1, cos1, sin2, cos2, sin3, cos3, *times_onehot], dtype=np.float32)


def decode_prediction_to_physical_np(pred_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_vec = np.asarray(pred_vec, dtype=np.float32)
    squeeze = False
    if pred_vec.ndim == 1:
        pred_vec = pred_vec[None, :]
        squeeze = True

    ia = np.clip(pred_vec[:, 0], 0.0, 1.0) * 90.0
    rad = np.clip(pred_vec[:, 1], 0.0, 1.0) * 200.0 + 50.0
    phi1 = np.mod(np.degrees(np.arctan2(pred_vec[:, 2], pred_vec[:, 3])), 360.0)
    phi2 = np.mod(np.degrees(np.arctan2(pred_vec[:, 4], pred_vec[:, 5])), 360.0)
    phi3 = np.mod(np.degrees(np.arctan2(pred_vec[:, 6], pred_vec[:, 7])), 360.0)
    times_idx = np.argmax(pred_vec[:, 8:11], axis=1)
    times = times_idx.astype(np.float32) + 1.0
    phi2[times < 2] = 0.0
    phi3[times < 3] = 0.0

    raw = np.stack([ia, phi1, rad, times, phi2, phi3], axis=1).astype(np.float32)
    norm = np.stack([
        ia / 90.0,
        phi1 / 360.0,
        (rad - 50.0) / 200.0,
        times / 3.0,
        phi2 / 360.0,
        phi3 / 360.0,
    ], axis=1).astype(np.float32)
    return (raw[0], norm[0]) if squeeze else (raw, norm)


def denormalize_label(label: Sequence[float]) -> Dict[str, float]:
    label = np.asarray(label, dtype=np.float32)
    times = float(np.clip(np.round(label[3] * 3.0), 1, 3))
    phi2 = float(label[4] * 360.0) if times >= 2 else 0.0
    phi3 = float(label[5] * 360.0) if times >= 3 else 0.0
    return {
        "ia": float(label[0] * 90.0),
        "phi1": float(label[1] * 360.0),
        "rad": float(label[2] * 200.0 + 50.0),
        "times": times,
        "phi2": phi2,
        "phi3": phi3,
    }


def project_candidate(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    x[0] = np.clip(x[0], 0.0, 1.0)
    x[1] = x[1] % 1.0
    x[2] = np.clip(x[2], 0.0, 1.0)
    times_int = int(np.clip(np.round(x[3] * 3.0), 1, 3))
    x[3] = times_int / 3.0
    x[4] = (x[4] % 1.0) if times_int >= 2 else 0.0
    x[5] = (x[5] % 1.0) if times_int >= 3 else 0.0
    return x


@dataclass
class SearchResult:
    method: str
    best_params: np.ndarray
    best_metrics: Dict[str, float]
    total_time_sec: float
    eval_count: int
    trace: List[Dict[str, float]]


def sample_random_candidate(rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros(6, dtype=np.float32)
    x[0] = rng.uniform(0.0, 1.0)
    x[1] = rng.uniform(0.0, 1.0)
    x[2] = rng.uniform(0.0, 1.0)
    x[3] = rng.choice([1 / 3, 2 / 3, 1.0])
    times = int(round(x[3] * 3))
    x[4] = rng.uniform(0.0, 1.0) if times >= 2 else 0.0
    x[5] = rng.uniform(0.0, 1.0) if times >= 3 else 0.0
    return project_candidate(x)


def evaluate_candidate(evaluator, target_img: torch.Tensor, true_params: np.ndarray, candidate: Sequence[float]) -> Dict[str, float]:
    return evaluator.evaluate(target_img, true_params, candidate)


def run_bicbam(initial_params: np.ndarray, evaluator, target_img: torch.Tensor, true_params: np.ndarray) -> SearchResult:
    start = time.perf_counter()
    metrics = evaluate_candidate(evaluator, target_img, true_params, initial_params)
    total = time.perf_counter() - start
    trace = [{"step": 1, "best_objective": metrics["objective"], "eval_count": 1, "elapsed_sec": total}]
    return SearchResult("Bi-CBAM", metrics["pred_params"], metrics, total, 1, trace)


def run_bicbam_refine(
    rng: np.random.RandomState,
    initial_params: np.ndarray,
    evaluator,
    target_img: torch.Tensor,
    true_params: np.ndarray,
    budget: int,
) -> SearchResult:
    start = time.perf_counter()
    current = project_candidate(initial_params)
    current_metrics = evaluate_candidate(evaluator, target_img, true_params, current)
    best_params = current.copy()
    best_metrics = current_metrics
    trace = [{"step": 1, "best_objective": best_metrics["objective"], "eval_count": 1, "elapsed_sec": time.perf_counter() - start}]
    eval_count = 1

    step_scales = np.array([0.05, 0.05, 0.05, 1 / 3, 0.05, 0.05], dtype=np.float32)
    while eval_count < budget:
        improved = False
        for dim in range(6):
            if eval_count >= budget:
                break
            for direction in (-1.0, 1.0):
                if eval_count >= budget:
                    break
                cand = current.copy()
                cand[dim] += direction * step_scales[dim]
                cand = project_candidate(cand)
                metrics = evaluate_candidate(evaluator, target_img, true_params, cand)
                eval_count += 1
                if metrics["objective"] < current_metrics["objective"]:
                    current = cand
                    current_metrics = metrics
                    improved = True
                if metrics["objective"] < best_metrics["objective"]:
                    best_metrics = metrics
                    best_params = cand.copy()
                trace.append({"step": eval_count, "best_objective": best_metrics["objective"], "eval_count": eval_count, "elapsed_sec": time.perf_counter() - start})
        if not improved:
            step_scales[:3] *= 0.6
            step_scales[4:] *= 0.6
            if np.max(step_scales[[0, 1, 2, 4, 5]]) < 0.01:
                while eval_count < budget:
                    cand = current + rng.normal(0.0, 0.02, size=current.shape).astype(np.float32)
                    cand = project_candidate(cand)
                    metrics = evaluate_candidate(evaluator, target_img, true_params, cand)
                    eval_count += 1
                    if metrics["objective"] < best_metrics["objective"]:
                        best_metrics = metrics
                        best_params = cand.copy()
                    trace.append({"step": eval_count, "best_objective": best_metrics["objective"], "eval_count": eval_count, "elapsed_sec": time.perf_counter() - start})
                break

    return SearchResult("Bi-CBAM+Refine", best_params, best_metrics, time.perf_counter() - start, eval_count, trace)


def build_samples(data_root: Path, excel_path: Path) -> List[Dict]:
    df = read_labels_from_excel(excel_path)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_map = {}
    for p in data_root.iterdir():
        if p.is_file() and p.suffix.lower() in valid_exts:
            image_map[p.stem.lower()] = p

    samples = []
    for row_idx, row in df.iterrows():
        stem = row["ImageStem"]
        if not stem:
            continue
        key = str(stem).lower()
        img_path = image_map.get(key)
        if img_path is None:
            for width in range(2, 7):
                img_path = image_map.get(key.zfill(width))
                if img_path is not None:
                    break
        if img_path is None:
            continue

        times = float(np.clip(float(row["Times"]), 1.0, 3.0))
        label = normalize_label(
            ia=float(row["IA"]),
            phi1=float(row["OADphi1"]),
            rad=float(row["SphereRad"]),
            times=times,
            phi2=float(row["OADphi2"]),
            phi3=float(row["OADphi3"]),
        )
        sample = {
            "row_index": int(row_idx),
            "image_name": str(row["ImageName"]),
            "image_path": str(img_path),
            "label": label,
            "times": int(round(times)),
        }
        samples.append(sample)
    if not samples:
        raise RuntimeError("No matched samples found.")
    return samples


class SampleDataset(data.Dataset):
    def __init__(self, samples: Sequence[Dict], transform=DATA_TRANSFORM):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        y = torch.tensor(sample["label"], dtype=torch.float32)
        return x, y


def split_samples(samples: Sequence[Dict], seed: int = SEED) -> Dict[str, List[Dict]]:
    rng = np.random.RandomState(seed)
    index = np.arange(len(samples))
    rng.shuffle(index)
    n = len(index)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    train_idx = index[:n_train]
    val_idx = index[n_train:n_train + n_val]
    test_idx = index[n_train + n_val:]
    items = list(samples)
    return {
        "train": [items[i] for i in train_idx],
        "val": [items[i] for i in val_idx],
        "test": [items[i] for i in test_idx],
    }


def image_complexity_score(image_tensor: torch.Tensor) -> float:
    x = image_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    gx = np.abs(np.diff(x, axis=1, prepend=x[:, :1]))
    gy = np.abs(np.diff(x, axis=0, prepend=x[:1, :]))
    edge_density = float((gx + gy).mean())
    fill_ratio = float((x > 0.5).mean())
    variance = float(x.var())
    return 0.5 * edge_density + 0.3 * fill_ratio + 0.2 * variance


def assign_difficulty_bins(samples: Sequence[Dict], transform=DATA_TRANSFORM) -> List[Dict]:
    enriched = []
    scores = []
    for sample in samples:
        with Image.open(sample["image_path"]) as img:
            img = img.convert("RGB")
            tensor = transform(img)
        score = image_complexity_score(tensor)
        scores.append(score)
        enriched.append({**sample, "complexity_score": score})
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    for item in enriched:
        score = item["complexity_score"]
        if score <= q1:
            difficulty = "low"
        elif score <= q2:
            difficulty = "mid"
        else:
            difficulty = "high"
        item["difficulty"] = difficulty
    return enriched


def stratified_sample(samples: Sequence[Dict], n_select: int, seed: int = SEED) -> List[Dict]:
    if n_select > len(samples):
        raise ValueError(f"Requested {n_select} samples, but only {len(samples)} available.")
    rng = np.random.RandomState(seed)
    enriched = assign_difficulty_bins(samples)
    groups: Dict[Tuple[int, str], List[Dict]] = {}
    for sample in enriched:
        key = (int(sample["times"]), sample["difficulty"])
        groups.setdefault(key, []).append(sample)

    keys = sorted(groups.keys())
    quotas = {key: n_select // len(keys) for key in keys}
    for key in keys[:n_select % len(keys)]:
        quotas[key] += 1

    chosen = []
    leftovers = []
    for key in keys:
        group = groups[key]
        rng.shuffle(group)
        take = min(len(group), quotas[key])
        chosen.extend(group[:take])
        leftovers.extend(group[take:])

    if len(chosen) < n_select:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:n_select - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:n_select]


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    if img2.dtype != img1.dtype:
        img2 = img2.to(dtype=img1.dtype)
    if img2.device != img1.device:
        img2 = img2.to(device=img1.device)
    if val_range is None:
        max_val = 1 if torch.max(img1) <= 128 else 255
        min_val = 0 if torch.min(img1) >= -0.5 else -1
        val_range = max_val - min_val
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device).type(img1.dtype)
    else:
        window = window.to(img1.device).type(img1.dtype)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )
        self.cbam = CBAM(out_ch, ratio=ratio)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )
        self.cbam = CBAM(out_ch, ratio=ratio)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.cbam(x)
        return x


class InverseCNN(nn.Module):
    def __init__(self, in_channels: int = 1, ratio: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            EncoderStage(in_channels, 16, ratio=ratio),
            EncoderStage(16, 32, ratio=ratio),
            EncoderStage(32, 64, ratio=ratio),
            EncoderStage(64, 96, ratio=ratio),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(84, 20),
            nn.ReLU(inplace=True),
        )
        self.head_cont = nn.Linear(20, 8)
        self.head_time = nn.Linear(20, 3)

    @staticmethod
    def _normalize_angle_pairs(angle_pairs):
        angle_pairs = torch.tanh(angle_pairs)
        reshaped = angle_pairs.view(angle_pairs.shape[0], 3, 2)
        norm = torch.norm(reshaped, dim=-1, keepdim=True).clamp_min(1e-6)
        reshaped = reshaped / norm
        return reshaped.view(angle_pairs.shape[0], 6)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.gap(feat)
        shared = self.shared(feat)
        raw_cont = self.head_cont(shared)
        times_logits = self.head_time(shared)
        ia_rad = torch.sigmoid(raw_cont[:, :2])
        angle_pairs = self._normalize_angle_pairs(raw_cont[:, 2:])
        return torch.cat([ia_rad, angle_pairs, times_logits], dim=1)


class ForwardCNN(nn.Module):
    def __init__(self, out_channels: int = 1, img_size=MODEL_INPUT_SIZE, ratio: int = 8):
        super().__init__()
        self.img_size = img_size
        self.seed_size = self._compute_seed_size(img_size)
        sh, sw = self.seed_size
        self.seed_channels = 96
        self.fc = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, self.seed_channels * sh * sw),
            nn.ReLU(inplace=True),
        )
        self.dec1 = DecoderStage(96, 64, ratio=ratio)
        self.dec2 = DecoderStage(64, 48, ratio=ratio)
        self.dec3 = DecoderStage(48, 32, ratio=ratio)
        self.dec4 = DecoderStage(32, 16, ratio=ratio)
        self.out_conv = nn.Sequential(
            ConvBNAct(16, 16, 3, 1),
            nn.Conv2d(16, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _compute_seed_size(img_size, levels: int = 4):
        h, w = img_size
        for _ in range(levels):
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
        return h, w

    def forward(self, packed_params):
        b = packed_params.shape[0]
        sh, sw = self.seed_size
        x = self.fc(packed_params)
        x = x.view(b, self.seed_channels, sh, sw)
        h, w = self.img_size
        sizes = [
            (math.ceil(h / 8), math.ceil(w / 8)),
            (math.ceil(h / 4), math.ceil(w / 4)),
            (math.ceil(h / 2), math.ceil(w / 2)),
            (h, w),
        ]
        x = self.dec1(x, sizes[0])
        x = self.dec2(x, sizes[1])
        x = self.dec3(x, sizes[2])
        x = self.dec4(x, sizes[3])
        return self.out_conv(x)


@dataclass
class LoadedModels:
    inverse: nn.Module
    forward: nn.Module
    device: torch.device


def load_models(
    model_dir: Path = PROJECT_ROOT.parent / "models",
    model_tag: Optional[str] = None,
    device: Optional[torch.device] = None,
    ratio: int = 8,
    kernel_size: int = 7,
) -> LoadedModels:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inverse = InverseCNN(in_channels=IMAGE_CHANNELS, ratio=ratio).to(device)
    forward = ForwardCNN(out_channels=IMAGE_CHANNELS, img_size=MODEL_INPUT_SIZE, ratio=ratio).to(device)
    train_state = model_dir / "train_state.pth"
    inv_pth = model_dir / "model_inverse.pth"
    fwd_pth = model_dir / "model_forward.pth"
    if train_state.exists():
        state = torch.load(str(train_state), map_location=device)
        inverse.load_state_dict(state["model_inverse_state_dict"])
        forward.load_state_dict(state["model_forward_state_dict"])
    elif inv_pth.exists() and fwd_pth.exists():
        inverse.load_state_dict(torch.load(str(inv_pth), map_location=device))
        forward.load_state_dict(torch.load(str(fwd_pth), map_location=device))
    elif model_tag is not None:
        train_state = model_dir / f"train_state-{model_tag}.pth"
        inv_pkl = model_dir / f"model_inverse-{model_tag}.pkl"
        fwd_pkl = model_dir / f"model_forward-{model_tag}.pkl"
        inv_pth_tag = model_dir / f"model_inverse-{model_tag}.pth"
        fwd_pth_tag = model_dir / f"model_forward-{model_tag}.pth"
        if train_state.exists():
            state = torch.load(str(train_state), map_location=device)
            inverse.load_state_dict(state["model_inverse_state_dict"])
            forward.load_state_dict(state["model_forward_state_dict"])
        elif inv_pth_tag.exists() and fwd_pth_tag.exists():
            inverse.load_state_dict(torch.load(str(inv_pth_tag), map_location=device))
            forward.load_state_dict(torch.load(str(fwd_pth_tag), map_location=device))
        elif inv_pkl.exists() and fwd_pkl.exists():
            inverse = torch.load(str(inv_pkl), map_location=device)
            forward = torch.load(str(fwd_pkl), map_location=device)
        else:
            raise FileNotFoundError("Could not find checkpoints in the model directory.")
    else:
        raise FileNotFoundError("Could not find checkpoints in the model directory.")
    inverse.eval()
    forward.eval()
    return LoadedModels(inverse=inverse, forward=forward, device=device)


@torch.no_grad()
def infer_params(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    pred = model(image_tensor.unsqueeze(0).to(device))
    _, pred_norm = decode_prediction_to_physical_np(pred.squeeze(0).detach().cpu().numpy())
    return project_candidate(pred_norm)


@torch.no_grad()
def render_from_params(model: nn.Module, params_norm: Sequence[float], device: torch.device) -> torch.Tensor:
    x = torch.tensor(pack_physical_norm6_np(params_norm), dtype=torch.float32, device=device).unsqueeze(0)
    out = model(x)
    return out.squeeze(0).detach().cpu()


def compute_metrics(target_img: torch.Tensor, pred_img: torch.Tensor, true_params: np.ndarray, pred_params: np.ndarray) -> Dict[str, float]:
    target = target_img.unsqueeze(0)
    pred = pred_img.unsqueeze(0)
    mse = float(F.mse_loss(pred, target).item())
    l1 = float(F.l1_loss(pred, target).item())
    ssim_val = float(ssim(pred, target).item())
    target_np = target_img.squeeze(0).detach().cpu().numpy().astype(np.float32)
    pred_np = pred_img.squeeze(0).detach().cpu().numpy().astype(np.float32)
    pcc = float(np.corrcoef(target_np.reshape(-1), pred_np.reshape(-1))[0, 1]) if target_np.std() > 0 and pred_np.std() > 0 else 0.0
    param_mae = float(np.mean(np.abs(true_params - pred_params)))
    objective = 0.5 * mse + 0.5 * (1.0 - ssim_val)
    return {
        "objective": objective,
        "mse": mse,
        "l1": l1,
        "ssim": ssim_val,
        "pcc": pcc,
        "param_mae": param_mae,
    }


class ForwardEvaluator:
    def __init__(self, forward_model: nn.Module, device: torch.device):
        self.forward_model = forward_model
        self.device = device
        self.calls = 0

    def evaluate(self, target_img: torch.Tensor, true_params: np.ndarray, params_norm: Sequence[float]) -> Dict[str, float]:
        params_norm = project_candidate(params_norm)
        start = time.perf_counter()
        pred_img = render_from_params(self.forward_model, params_norm, self.device)
        elapsed = time.perf_counter() - start
        self.calls += 1
        metrics = compute_metrics(target_img, pred_img, true_params, params_norm)
        metrics["eval_time_sec"] = elapsed
        metrics["pred_img"] = pred_img
        metrics["pred_params"] = params_norm
        return metrics

