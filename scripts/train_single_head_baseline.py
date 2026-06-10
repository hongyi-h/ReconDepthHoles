"""
Phase 0.5 single-head composite-target baseline.

This baseline tests the main confound after Gate 1:
  Can a non-layered, single corrected pointmap head learn the same synthetic
  mirror target without explicitly predicting a secondary layer?

Target definition:
  - Lambertian pixels: first-surface pointmap GT
  - Non-Lambertian pixels with secondary GT: secondary-path pointmap GT

This is not the final desired representation; it is a hard control. If this
baseline matches the layered secondary head on NL pixels while preserving
Lambertian pixels, the current evidence does not prove the layered
representation is necessary.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# MetaX C500 PyTorch builds can trip over torch._dynamo -> triton backend
# discovery when optimizer code imports torch._compile. Disable compile/dynamo
# before importing torch; this baseline does not rely on graph compilation.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "third_party" / "vggt"))

from data.mirror_scene_dataset import MirrorSceneDataset, collate_fn
from vggt.heads.dpt_head import DPTHead
from vggt.models.vggt import VGGT


def choose_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SingleHeadVGGT(nn.Module):
    """Frozen VGGT encoder with one trainable corrected pointmap head."""

    def __init__(self, pretrained_name="facebook/VGGT-1B", freeze_encoder=True):
        super().__init__()
        self.vggt = VGGT.from_pretrained(pretrained_name)

        if freeze_encoder:
            for param in self.vggt.aggregator.parameters():
                param.requires_grad = False
            for head in [self.vggt.point_head, self.vggt.depth_head, self.vggt.camera_head]:
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad = False

        embed_dim = 1024
        try:
            self.corrected_point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            self.mask_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="sigmoid",
                conf_activation="expp1",
            )
        except TypeError:
            self.corrected_point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
            )
            self.mask_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="sigmoid",
            )

        self._print_param_stats()

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SingleHeadVGGT] Total params: {total/1e6:.1f}M")
        print(f"[SingleHeadVGGT] Trainable params: {trainable/1e6:.1f}M")
        print(f"[SingleHeadVGGT] Frozen params: {(total-trainable)/1e6:.1f}M")

    def forward(self, images):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)

        predictions = {}
        with torch.amp.autocast("cuda", enabled=False):
            if self.vggt.point_head is not None:
                with torch.no_grad():
                    pts3d, pts3d_conf = self.vggt.point_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                predictions["vggt_world_points"] = pts3d
                predictions["vggt_world_points_conf"] = pts3d_conf

            corrected, corrected_conf = self.corrected_point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            mask_pred, mask_conf = self.mask_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )

        predictions["corrected_points"] = corrected
        predictions["corrected_points_conf"] = corrected_conf
        predictions["mask_pred"] = mask_pred.squeeze(-1)
        predictions["mask_conf"] = mask_conf
        return predictions


class SingleHeadLoss(nn.Module):
    def __init__(self, lambda_mask=0.3):
        super().__init__()
        self.lambda_mask = lambda_mask

    def forward(self, predictions, batch):
        gt_first = batch["world_points"]
        gt_secondary = batch["world_points_secondary"]
        mask_first = batch["point_masks"]
        mask_secondary = batch["point_masks_secondary"]
        pred = predictions["corrected_points"]

        composite_target = torch.where(mask_secondary.unsqueeze(-1), gt_secondary, gt_first)
        if mask_first.sum() > 0:
            point_loss = (pred - composite_target).abs()[mask_first].mean()
        else:
            point_loss = torch.tensor(0.0, device=pred.device)

        pred_mask = predictions["mask_pred"]
        gt_mask = mask_secondary.float()
        if mask_first.sum() > 0:
            mask_loss = F.binary_cross_entropy(pred_mask[mask_first], gt_mask[mask_first])
        else:
            mask_loss = torch.tensor(0.0, device=pred.device)

        total = point_loss + self.lambda_mask * mask_loss
        return {
            "loss_total": total,
            "loss_point": point_loss,
            "loss_mask": mask_loss,
        }


class MetricAccumulator:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add_l2(self, pred, gt, mask):
        if mask.sum().item() == 0:
            return
        diff = (pred - gt).norm(dim=-1)
        values = diff[mask]
        self.total += values.sum().item()
        self.count += values.numel()

    def add_binary_acc(self, pred_binary, gt_binary, mask):
        if mask.sum().item() == 0:
            return
        values = (pred_binary[mask] == gt_binary[mask]).float()
        self.total += values.sum().item()
        self.count += values.numel()

    def mean(self):
        return self.total / self.count if self.count else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, mask_threshold=0.5):
    model.eval()
    metrics = {
        "vggt_primary_vs_first_l": MetricAccumulator(),
        "vggt_primary_vs_secondary_nl": MetricAccumulator(),
        "single_corrected_vs_first_l": MetricAccumulator(),
        "single_corrected_vs_secondary_nl": MetricAccumulator(),
        "single_corrected_vs_first_nl": MetricAccumulator(),
        "single_mask_acc_valid_first": MetricAccumulator(),
    }

    for batch in loader:
        images = batch["images"].to(device)
        gt_first = batch["world_points"].to(device)
        gt_secondary = batch["world_points_secondary"].to(device)
        mask_first = batch["point_masks"].to(device)
        mask_secondary = batch["point_masks_secondary"].to(device)
        mask_l = mask_first & ~mask_secondary
        mask_nl = mask_first & mask_secondary

        pred = model(images)
        vggt_first = pred["vggt_world_points"]
        corrected = pred["corrected_points"]
        pred_mask = pred["mask_pred"] > mask_threshold

        metrics["vggt_primary_vs_first_l"].add_l2(vggt_first, gt_first, mask_l)
        metrics["vggt_primary_vs_secondary_nl"].add_l2(vggt_first, gt_secondary, mask_nl)
        metrics["single_corrected_vs_first_l"].add_l2(corrected, gt_first, mask_l)
        metrics["single_corrected_vs_secondary_nl"].add_l2(corrected, gt_secondary, mask_nl)
        metrics["single_corrected_vs_first_nl"].add_l2(corrected, gt_first, mask_nl)
        metrics["single_mask_acc_valid_first"].add_binary_acc(pred_mask, mask_secondary, mask_first)

    model.train()
    return {
        "metrics": {key: value.mean() for key, value in metrics.items()},
        "counts": {key: value.count for key, value in metrics.items()},
    }


def make_loader(data_dir, args, shuffle):
    dataset = MirrorSceneDataset(
        data_dir,
        num_views=args.num_views,
        img_size=args.img_size,
        max_scenes=args.max_scenes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
        drop_last=shuffle,
    )
    return dataset, loader


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_dataset, train_loader = make_loader(args.data_dir, args, shuffle=True)
    val_dataset, val_loader = make_loader(args.val_dir, args, shuffle=False)

    model = SingleHeadVGGT(args.pretrained_name, freeze_encoder=True).to(device)
    criterion = SingleHeadLoss(lambda_mask=args.lambda_mask)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * len(train_loader), 1),
        eta_min=args.lr * 0.01,
    )

    print("=" * 60)
    print("Starting Single-Head Composite Baseline")
    print(f"  Train scenes: {len(train_dataset)}")
    print(f"  Val scenes: {len(val_dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {device}")
    print("=" * 60)

    latest_results = None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            for key in [
                "world_points",
                "world_points_secondary",
                "point_masks",
                "point_masks_secondary",
            ]:
                batch[key] = batch[key].to(device)

            pred = model(images)
            loss_dict = criterion(pred, batch)

            optimizer.zero_grad()
            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss_dict["loss_total"].item())

            if (step + 1) % args.log_every == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {np.mean(losses[-args.log_every:]):.4f} | "
                    f"L_point: {loss_dict['loss_point'].item():.4f} | "
                    f"L_mask: {loss_dict['loss_mask'].item():.4f}"
                )

        print(f"Epoch {epoch+1} done in {time.time()-t0:.0f}s | Avg loss: {np.mean(losses):.4f}")

        if (epoch + 1) % args.eval_every == 0:
            latest_results = evaluate(model, val_loader, device, args.mask_threshold)
            print(json.dumps({"epoch": epoch + 1, **latest_results}, indent=2))

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, np.mean(losses), args)

    if latest_results is None:
        latest_results = evaluate(model, val_loader, device, args.mask_threshold)

    save_checkpoint(model, optimizer, args.epochs - 1, float("nan"), args, final=True)
    write_outputs(args, latest_results, len(train_dataset), len(val_dataset), str(device))
    print(json.dumps(latest_results, indent=2))


def save_checkpoint(model, optimizer, epoch, loss, args, final=False):
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "args": vars(args),
    }
    name = "single_head_final.pt" if final else f"single_head_epoch{epoch+1:03d}.pt"
    torch.save(state, ckpt_dir / name)
    torch.save(state, ckpt_dir / "single_head_latest.pt")
    print(f"Saved checkpoint: {ckpt_dir / name}")


def write_outputs(args, results, train_scenes, val_scenes, device):
    output = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "train_dir": args.data_dir,
        "val_dir": args.val_dir,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "num_views": args.num_views,
        "epochs": args.epochs,
        "device": device,
        **results,
    }
    base = output["metrics"]["vggt_primary_vs_secondary_nl"]
    single = output["metrics"]["single_corrected_vs_secondary_nl"]
    if np.isfinite(base) and np.isfinite(single) and base > 0:
        output["derived"] = {
            "single_head_reduction_vs_vggt_primary_to_secondary": (base - single) / base,
        }

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.output_note:
        write_note(Path(args.output_note), output)


def write_note(path, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "tags: [experiment, phase-0-5, single-head, baseline]",
        f"created: {output['created']}",
        "---",
        "",
        "# Phase 0.5 Single-Head Baseline",
        "",
        "## Setup",
        "",
        f"- Train data: `{output['train_dir']}`",
        f"- Val data: `{output['val_dir']}`",
        f"- Train scenes: `{output['train_scenes']}`",
        f"- Val scenes: `{output['val_scenes']}`",
        f"- Views per scene: `{output['num_views']}`",
        f"- Epochs: `{output['epochs']}`",
        f"- Device: `{output['device']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Count |",
        "|---|---:|---:|",
    ]
    for key, value in output["metrics"].items():
        lines.append(f"| `{key}` | {value:.6f} | {output['counts'][key]} |")
    if "derived" in output:
        lines.extend(["", "## Derived"])
        for key, value in output["derived"].items():
            lines.append(f"- `{key}`: {value:.6f}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This baseline predicts one composite pointmap: first-surface GT on Lambertian pixels and secondary-path GT on non-Lambertian pixels. It tests whether a non-layered corrected pointmap can match the layered secondary head.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train single-head composite baseline.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic/train")
    parser.add_argument("--val_dir", type=str, default="data/synthetic/val")
    parser.add_argument("--pretrained_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_mask", type=float, default=0.3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phase05_single_head")
    parser.add_argument("--output_json", type=str, default="notes/20-Experiments/phase05_single_head_baseline.json")
    parser.add_argument("--output_note", type=str, default="notes/20-Experiments/Phase-0.5-Single-Head-Baseline.md")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
