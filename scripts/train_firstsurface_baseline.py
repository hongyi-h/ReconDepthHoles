"""
Phase 0.5 first-surface-only multitask baseline (clean control for Gate 3).

Why this exists
---------------
Gate 3 claimed the dual-head layered model beats the single-head composite
baseline on first-surface recovery at NL pixels by 81% (0.268871 vs 1.436062).
That comparison is a tautology: the single-head was supervised on the COMPOSITE
target (secondary GT on NL pixels), so of course its NL output is far from
first-surface GT.

This script provides the honest control the single-head could not:
  a model that predicts ONLY first-surface (+ NL mask), supervised on
  first-surface GT over all valid pixels, with everything else held identical
  to the dual-head baseline (frozen encoder, same DPTHead config, same
  optimizer / schedule / grad-clip / seed / data / epochs).

What it answers
---------------
  - The honest first-surface reference: how low can NL first-surface error go
    with a dedicated first-surface head? -> firstonly_vs_first_nl
  - Is the dual-head's secondary head FREE on the first-surface task?
    Compare firstonly_vs_first_nl against dual_primary_vs_first_nl = 0.268871
    and firstonly_vs_first_l against dual_primary_vs_first_l = 0.311777.

What it does NOT answer
-----------------------
  Whether the layered architecture is *necessary*. With a frozen encoder the
  primary/secondary/mask heads are independent probes that cannot interfere by
  construction; that question is only meaningful under an unfrozen encoder
  (Phase 1). See notes/00-Project/Thesis-Decision-2026-06-12.md.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# MetaX C500 compatibility: avoid torch._dynamo -> Triton backend discovery.
# (The single-head launch failed here until this guard was added; keep it.)
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


class FirstSurfaceVGGT(nn.Module):
    """Frozen VGGT encoder with trainable first-surface head + NL mask head.

    Identical to DualHeadVGGT minus the secondary point head. This is the
    dedicated first-surface control: same frozen tokens, same head config.
    """

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
            self.primary_point_head = DPTHead(
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
            self.primary_point_head = DPTHead(
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
        print(f"[FirstSurfaceVGGT] Total params: {total/1e6:.1f}M")
        print(f"[FirstSurfaceVGGT] Trainable params: {trainable/1e6:.1f}M")
        print(f"[FirstSurfaceVGGT] Frozen params: {(total-trainable)/1e6:.1f}M")

    def forward(self, images):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)

        predictions = {}
        with torch.amp.autocast("cuda", enabled=False):
            if self.vggt.point_head is not None:
                with torch.no_grad():
                    vggt_points, vggt_conf = self.vggt.point_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                predictions["vggt_world_points"] = vggt_points
                predictions["vggt_world_points_conf"] = vggt_conf

            primary, primary_conf = self.primary_point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            mask_pred, mask_conf = self.mask_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )

        predictions["primary_points"] = primary
        predictions["primary_points_conf"] = primary_conf
        predictions["mask_pred"] = mask_pred.squeeze(-1)
        predictions["mask_conf"] = mask_conf
        return predictions


class FirstSurfaceLoss(nn.Module):
    """First-surface point loss on all valid pixels + NL mask BCE.

    Mirrors the primary + mask terms of DualHeadLoss exactly (no secondary).
    """

    def __init__(self, lambda_primary=1.0, lambda_mask=0.3):
        super().__init__()
        self.lambda_primary = lambda_primary
        self.lambda_mask = lambda_mask

    def forward(self, predictions, batch):
        gt_first = batch["world_points"]
        mask_first = batch["point_masks"]
        mask_secondary = batch["point_masks_secondary"]

        primary = predictions["primary_points"]

        if mask_first.sum() > 0:
            loss_primary = (primary - gt_first).abs()[mask_first].mean()
        else:
            loss_primary = torch.tensor(0.0, device=primary.device)

        pred_mask = predictions["mask_pred"]
        gt_mask = mask_secondary.float()
        if mask_first.sum() > 0:
            loss_mask = F.binary_cross_entropy(pred_mask[mask_first], gt_mask[mask_first])
        else:
            loss_mask = torch.tensor(0.0, device=primary.device)

        total = self.lambda_primary * loss_primary + self.lambda_mask * loss_mask
        return {
            "loss_total": total,
            "loss_primary": loss_primary,
            "loss_mask": loss_mask,
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
        "vggt_primary_vs_first_nl": MetricAccumulator(),
        "firstonly_vs_first_l": MetricAccumulator(),
        "firstonly_vs_first_nl": MetricAccumulator(),
        "firstonly_mask_acc_valid_first": MetricAccumulator(),
    }

    for batch in loader:
        images = batch["images"].to(device)
        gt_first = batch["world_points"].to(device)
        mask_first = batch["point_masks"].to(device)
        mask_secondary = batch["point_masks_secondary"].to(device)
        mask_l = mask_first & ~mask_secondary
        mask_nl = mask_first & mask_secondary

        pred = model(images)
        vggt_first = pred["vggt_world_points"]
        primary = pred["primary_points"]
        pred_mask = pred["mask_pred"] > mask_threshold

        metrics["vggt_primary_vs_first_l"].add_l2(vggt_first, gt_first, mask_l)
        metrics["vggt_primary_vs_first_nl"].add_l2(vggt_first, gt_first, mask_nl)
        metrics["firstonly_vs_first_l"].add_l2(primary, gt_first, mask_l)
        metrics["firstonly_vs_first_nl"].add_l2(primary, gt_first, mask_nl)
        metrics["firstonly_mask_acc_valid_first"].add_binary_acc(pred_mask, mask_secondary, mask_first)

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

    model = FirstSurfaceVGGT(args.pretrained_name, freeze_encoder=True).to(device)
    criterion = FirstSurfaceLoss(
        lambda_primary=args.lambda_primary,
        lambda_mask=args.lambda_mask,
    )
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
    print("Starting First-Surface-Only Baseline (clean control for Gate 3)")
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
            for key in ["world_points", "point_masks", "point_masks_secondary"]:
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
                    f"L_primary: {loss_dict['loss_primary'].item():.4f} | "
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
    name = "firstonly_final.pt" if final else f"firstonly_epoch{epoch+1:03d}.pt"
    torch.save(state, ckpt_dir / name)
    torch.save(state, ckpt_dir / "firstonly_latest.pt")
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

    # Direct comparison against the dual-head primary head (the layered model's
    # first-surface task) to test whether the secondary head is free.
    ref = {
        "dual_primary_vs_first_nl": 0.268871,
        "dual_primary_vs_first_l": 0.311777,
    }
    fo_nl = output["metrics"].get("firstonly_vs_first_nl")
    fo_l = output["metrics"].get("firstonly_vs_first_l")
    derived = {"reference_dual_head": ref}
    if fo_nl is not None and np.isfinite(fo_nl):
        derived["firstonly_minus_dual_primary_nl"] = fo_nl - ref["dual_primary_vs_first_nl"]
        derived["firstonly_over_dual_primary_nl_ratio"] = fo_nl / ref["dual_primary_vs_first_nl"]
    if fo_l is not None and np.isfinite(fo_l):
        derived["firstonly_minus_dual_primary_l"] = fo_l - ref["dual_primary_vs_first_l"]
    output["derived"] = derived

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
        "tags: [experiment, phase-0-5, first-surface, baseline, clean-control]",
        f"created: {output['created']}",
        "---",
        "",
        "# Phase 0.5 First-Surface-Only Baseline (Clean Control)",
        "",
        "## Purpose",
        "",
        "Honest first-surface control for Gate 3. Replaces the tautological "
        "single-head first-layer comparison (single-head was supervised on the "
        "composite target, so its NL output approximates secondary, not first). "
        "This model predicts ONLY first-surface (+ NL mask) over all valid pixels, "
        "everything else identical to the dual-head baseline.",
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
    lines.extend(["", "## Comparison vs Dual-Head Primary (is the secondary head free?)", ""])
    d = output.get("derived", {})
    ref = d.get("reference_dual_head", {})
    lines.append(f"- dual_primary_vs_first_nl (reference): `{ref.get('dual_primary_vs_first_nl')}`")
    lines.append(f"- dual_primary_vs_first_l (reference): `{ref.get('dual_primary_vs_first_l')}`")
    if "firstonly_over_dual_primary_nl_ratio" in d:
        lines.append(f"- firstonly / dual_primary NL ratio: `{d['firstonly_over_dual_primary_nl_ratio']:.4f}`")
    if "firstonly_minus_dual_primary_nl" in d:
        lines.append(f"- firstonly - dual_primary NL: `{d['firstonly_minus_dual_primary_nl']:.6f}`")
    if "firstonly_minus_dual_primary_l" in d:
        lines.append(f"- firstonly - dual_primary L: `{d['firstonly_minus_dual_primary_l']:.6f}`")
    lines.extend([
        "",
        "## Decision Rule",
        "",
        "- If `firstonly_vs_first_nl` is approximately equal to the dual-head primary "
        "(`0.268871`), the secondary head is FREE on the first-surface task: the "
        "layered model recovers first-surface as well as a dedicated first-surface "
        "model AND additionally recovers secondary geometry. This supports the "
        "Thesis-B free-lunch framing.",
        "- If dual-head primary is clearly WORSE than firstonly, the secondary head "
        "hurts first-surface via shared optimizer / global grad-clip coupling; report "
        "this as a real cost and consider decoupled optimization.",
        "- If dual-head primary is clearly BETTER, the secondary task acts as a "
        "regularizer for first-surface; an interesting positive-transfer result.",
        "",
        "## Scope Limit",
        "",
        "With a frozen encoder the heads are independent probes and cannot interfere "
        "by construction. This baseline does NOT establish that the layered "
        "architecture is necessary; that question is only meaningful under an "
        "unfrozen encoder (Phase 1). See `notes/00-Project/Thesis-Decision-2026-06-12.md`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train first-surface-only clean control baseline.")
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
    parser.add_argument("--lambda_primary", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=0.3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phase05_firstonly")
    parser.add_argument("--output_json", type=str, default="notes/20-Experiments/phase05_firstonly_baseline.json")
    parser.add_argument("--output_note", type=str, default="notes/20-Experiments/Phase-0.5-FirstSurface-Baseline.md")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
