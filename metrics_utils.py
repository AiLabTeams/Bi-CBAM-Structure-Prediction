import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    if img2.dtype != img1.dtype:
        img2 = img2.to(dtype=img1.dtype)
    if img2.device != img1.device:
        img2 = img2.to(device=img1.device)

    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        dynamic_range = max_val - min_val
    else:
        dynamic_range = val_range

    (_, channel, height, width) = img1.size()
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
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def pack_prediction_for_forward(pred):
    times_probs = F.softmax(pred[:, 8:11], dim=1)
    return torch.cat([pred[:, :8], times_probs], dim=1)


def circular_deg_from_pair_np(sin_arr, cos_arr):
    ang = np.degrees(np.arctan2(sin_arr, cos_arr))
    return np.mod(ang, 360.0)


def decode_prediction_to_physical_np(pred_vec):
    pred_vec = np.asarray(pred_vec, dtype=np.float32)
    ia = np.clip(pred_vec[:, 0], 0.0, 1.0) * 90.0
    rad = np.clip(pred_vec[:, 1], 0.0, 1.0) * 200.0 + 50.0
    phi1 = circular_deg_from_pair_np(pred_vec[:, 2], pred_vec[:, 3])
    phi2 = circular_deg_from_pair_np(pred_vec[:, 4], pred_vec[:, 5])
    phi3 = circular_deg_from_pair_np(pred_vec[:, 6], pred_vec[:, 7])
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
    return raw, norm


def decode_prediction_to_physical_norm6_torch(pred_param):
    def circular_norm_from_pair_torch(sin_comp, cos_comp):
        ang = torch.atan2(sin_comp, cos_comp)
        ang = torch.remainder(ang, 2.0 * math.pi)
        return ang / (2.0 * math.pi)

    ia = pred_param[:, 0].clamp(0.0, 1.0)
    rad = pred_param[:, 1].clamp(0.0, 1.0)
    phi1 = circular_norm_from_pair_torch(pred_param[:, 2], pred_param[:, 3])
    phi2 = circular_norm_from_pair_torch(pred_param[:, 4], pred_param[:, 5])
    phi3 = circular_norm_from_pair_torch(pred_param[:, 6], pred_param[:, 7])
    times_probs = F.softmax(pred_param[:, 8:11], dim=1)
    time_levels = pred_param.new_tensor([1.0 / 3.0, 2.0 / 3.0, 1.0])
    times_norm = torch.sum(times_probs * time_levels.unsqueeze(0), dim=1)
    mask_phi2 = times_probs[:, 1] + times_probs[:, 2]
    mask_phi3 = times_probs[:, 2]
    pred_norm6 = torch.stack([ia, phi1, rad, times_norm, phi2 * mask_phi2, phi3 * mask_phi3], dim=1)
    return pred_norm6, times_probs


def compute_samplewise_image_metrics(target_imgs: torch.Tensor, pred_imgs: torch.Tensor):
    target_np = target_imgs.detach().cpu().numpy().astype(np.float32)
    pred_np = pred_imgs.detach().cpu().numpy().astype(np.float32)
    pcc_list = []
    ssim_list = []
    for i in range(target_np.shape[0]):
        tgt = target_np[i].reshape(-1)
        pred = pred_np[i].reshape(-1)
        if float(np.std(tgt)) > 0.0 and float(np.std(pred)) > 0.0:
            pcc = float(np.corrcoef(tgt, pred)[0, 1])
        else:
            pcc = 0.0
        tgt_t = torch.from_numpy(target_np[i:i + 1])
        pred_t = torch.from_numpy(pred_np[i:i + 1])
        ssim_val = float(ssim(pred_t, tgt_t, size_average=True).item())
        pcc_list.append(pcc)
        ssim_list.append(ssim_val)
    return np.asarray(pcc_list, dtype=np.float32), np.asarray(ssim_list, dtype=np.float32)


def bootstrap_confidence_interval(values, n_boot=1000, alpha=0.05, seed=42):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def build_test_metrics_summary(target_imgs, pred_imgs, true_norm6_np, pred_norm6_np, seed):
    pcc_vals, ssim_vals = compute_samplewise_image_metrics(target_imgs, pred_imgs)
    summary = {
        "n_test": int(len(pcc_vals)),
        "pcc_mean": float(np.mean(pcc_vals)),
        "pcc_std": float(np.std(pcc_vals, ddof=1)) if len(pcc_vals) > 1 else 0.0,
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals, ddof=1)) if len(ssim_vals) > 1 else 0.0,
        "param_r2_overall": float(r2_score(true_norm6_np, pred_norm6_np)),
    }
    summary["pcc_ci95_low"], summary["pcc_ci95_high"] = bootstrap_confidence_interval(pcc_vals, seed=seed)
    summary["ssim_ci95_low"], summary["ssim_ci95_high"] = bootstrap_confidence_interval(ssim_vals, seed=seed)
    for idx, name in enumerate(["ia", "phi1", "rad", "times", "phi2", "phi3"]):
        try:
            summary[f"r2_{name}"] = float(r2_score(true_norm6_np[:, idx], pred_norm6_np[:, idx]))
        except Exception:
            summary[f"r2_{name}"] = np.nan
    return summary, pcc_vals, ssim_vals
