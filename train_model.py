import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data

from bicbam_network import ForwardBiCBAMNetV2, InverseBiCBAMNetV2
from data_utils import DATA_TRANSFORM, LazyImageLabelDataset, build_samples, split_indices
from metrics_utils import SSIM, decode_prediction_to_physical_norm6_torch, pack_prediction_for_forward


IMAGE_CHANNELS = 1
SOURCE_IMAGE_SIZE = (138, 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the inverse-forward model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--excel-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs/train_repro"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--cls-weight", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class BiCBAMLossV2(torch.nn.Module):
    def __init__(self, ssim_fn, alpha=0.5, beta=0.5, cls_weight=0.35):
        super().__init__()
        self.ssim_fn = ssim_fn
        self.alpha = alpha
        self.beta = beta
        self.cls_weight = cls_weight
        self.image_mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pred_param, target, pred_img, true_img):
        pred_time_logits = pred_param[:, 8:11]
        pred_norm6, _ = decode_prediction_to_physical_norm6_torch(pred_param)
        true_norm6 = target["physical_norm6"]
        gt_times = target["times_idx"]
        loss_inv = torch.mean(torch.abs(pred_norm6 - true_norm6))
        loss_cls = self.ce(pred_time_logits, gt_times)
        loss_img = 0.5 * self.image_mse(pred_img, true_img) + 0.5 * (1.0 - self.ssim_fn(pred_img, true_img))
        loss_all = self.alpha * loss_inv + self.beta * loss_img + self.cls_weight * loss_cls
        return loss_all, loss_inv, loss_img, loss_cls


def move_target_to_device(target, device):
    return {k: v.to(device, non_blocking=True) for k, v in target.items()}


def evaluate(model_inverse, model_forward, loader, criterion, device):
    model_inverse.eval()
    model_forward.eval()
    val_loss_sum = 0.0
    val_count = 0
    val_correct_times = 0
    with torch.no_grad():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = move_target_to_device(target, device)
            pred = model_inverse(x)
            recon = model_forward(pack_prediction_for_forward(pred))
            loss_all, _, _, _ = criterion(pred, target, recon, x)
            bs = x.size(0)
            val_loss_sum += float(loss_all.item()) * bs
            val_count += bs
            pred_time_idx = torch.argmax(pred[:, 8:11], dim=1)
            val_correct_times += int((pred_time_idx == target["times_idx"]).sum().item())
    return val_loss_sum / max(val_count, 1), val_correct_times / max(val_count, 1)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (Path(__file__).resolve().parent / args.output_dir).resolve()
    model_dir = output_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    samples = build_samples(args.data_root, args.excel_path)
    train_idx, val_idx, _ = split_indices(len(samples), seed=args.seed)
    train_set = LazyImageLabelDataset([samples[i] for i in train_idx], transform=DATA_TRANSFORM)
    val_set = LazyImageLabelDataset([samples[i] for i in val_idx], transform=DATA_TRANSFORM)

    train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = Data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_inverse = InverseBiCBAMNetV2(in_channels=IMAGE_CHANNELS, ratio=8).to(device)
    model_forward = ForwardBiCBAMNetV2(out_channels=IMAGE_CHANNELS, img_size=SOURCE_IMAGE_SIZE, ratio=8).to(device)
    criterion = BiCBAMLossV2(SSIM(window_size=11).to(device), alpha=args.alpha, beta=args.beta, cls_weight=args.cls_weight)
    opt1 = torch.optim.Adam(model_inverse.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt2 = torch.optim.Adam(model_forward.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model_inverse.train()
        model_forward.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, target in train_loader:
            x = x.to(device, non_blocking=True)
            target = move_target_to_device(target, device)
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            pred = model_inverse(x)
            recon = model_forward(pack_prediction_for_forward(pred))
            loss_all, _, _, _ = criterion(pred, target, recon, x)
            loss_all.backward()
            opt1.step()
            opt2.step()
            bs = x.size(0)
            train_loss_sum += float(loss_all.item()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss, val_time_acc = evaluate(model_inverse, model_forward, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_time_acc": val_time_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model_inverse.state_dict(), model_dir / "model_inverse.pth")
            torch.save(model_forward.state_dict(), model_dir / "model_forward.pth")
        else:
            patience_counter += 1

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_time_acc={val_time_acc:.4f}")
        if patience_counter >= args.patience:
            break

    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

