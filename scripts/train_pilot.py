"""
Pilot Training Script for Layered Pointmap Reconstruction.

Strategy:
  1. Load VGGT pretrained (1.26B params)
  2. Freeze aggregator (encoder) completely
  3. Keep original point_head frozen (as baseline reference)
  4. Add NEW secondary_point_head (same architecture as point_head)
  5. Add NEW mask_head (predicts Lambertian/non-Lambertian per pixel)
  6. Train only new heads on synthetic mirror data
  7. Evaluate: Chamfer-NL vs VGGT zero-shot

Usage:
  # Single GPU (debug):
  python scripts/train_pilot.py --data_dir data/synthetic/train --val_dir data/synthetic/val --gpus 1

  # Multi-GPU (C500):
  torchrun --nproc_per_node=4 scripts/train_pilot.py \
    --data_dir data/synthetic/train --val_dir data/synthetic/val --gpus 4

  # Resume:
  python scripts/train_pilot.py --resume checkpoints/pilot_latest.pt
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# Disable torch dynamo/compile (incompatible with MetaX PyTorch build)
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Add project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "third_party" / "vggt"))

from vggt.models.vggt import VGGT
from vggt.heads.dpt_head import DPTHead
from data.mirror_scene_dataset import MirrorSceneDataset, collate_fn


# ============================================================
# Model: VGGT + Layered Heads
# ============================================================

class LayeredVGGT(nn.Module):
    """VGGT with additional heads for secondary-path pointmap and material mask."""

    def __init__(self, pretrained_name="facebook/VGGT-1B", freeze_encoder=True):
        super().__init__()

        # Load pretrained VGGT
        self.vggt = VGGT.from_pretrained(pretrained_name)

        # Freeze encoder (aggregator) and original heads
        if freeze_encoder:
            for param in self.vggt.aggregator.parameters():
                param.requires_grad = False
            # Also freeze original heads (we keep them for reference / loss on first-surface)
            if self.vggt.point_head is not None:
                for param in self.vggt.point_head.parameters():
                    param.requires_grad = False
            if self.vggt.depth_head is not None:
                for param in self.vggt.depth_head.parameters():
                    param.requires_grad = False
            if self.vggt.camera_head is not None:
                for param in self.vggt.camera_head.parameters():
                    param.requires_grad = False

        # NEW: Secondary-path pointmap head (same architecture as point_head)
        embed_dim = 1024  # VGGT-1B embed_dim
        try:
            self.secondary_point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,  # 3 for XYZ + 1 for confidence
                activation="inv_log",
                conf_activation="expp1",
            )
        except TypeError:
            # Older VGGT version without conf_activation param
            self.secondary_point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
            )

        # NEW: Material mask head (binary: Lambertian vs non-Lambertian)
        # output_dim=2: 1 channel for mask + 1 for confidence (DPTHead splits last as conf)
        try:
            self.mask_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="sigmoid",
                conf_activation="expp1",
            )
        except TypeError:
            self.mask_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="sigmoid",
            )

        self._print_param_stats()

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LayeredVGGT] Total params: {total/1e6:.1f}M")
        print(f"[LayeredVGGT] Trainable params: {trainable/1e6:.1f}M")
        print(f"[LayeredVGGT] Frozen params: {(total-trainable)/1e6:.1f}M")

    def forward(self, images):
        """
        Args:
            images: (B, S, 3, H, W) or (S, 3, H, W)

        Returns:
            dict with:
                - world_points: (B, S, H, W, 3) — first-surface (from frozen head)
                - world_points_conf: (B, S, H, W)
                - secondary_points: (B, S, H, W, 3) — secondary-path (from new head)
                - secondary_points_conf: (B, S, H, W)
                - mask_pred: (B, S, H, W) — non-Lambertian probability
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # Shared encoder pass (frozen)
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)

        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            # Original point head (frozen, for first-surface reference)
            if self.vggt.point_head is not None:
                with torch.no_grad():
                    pts3d, pts3d_conf = self.vggt.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            # NEW: Secondary-path pointmap head (trainable)
            secondary_pts, secondary_conf = self.secondary_point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["secondary_points"] = secondary_pts
            predictions["secondary_points_conf"] = secondary_conf

            # NEW: Material mask head (trainable)
            mask_pred, mask_conf = self.mask_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            # mask_pred shape: (B, S, H, W, 1) — squeeze to (B, S, H, W)
            predictions["mask_pred"] = mask_pred.squeeze(-1)
            predictions["mask_conf"] = mask_conf

        return predictions


# ============================================================
# Loss
# ============================================================

class LayeredPointmapLoss(nn.Module):
    """Loss for layered pointmap training."""

    def __init__(self, lambda_secondary=1.0, lambda_mask=0.3, lambda_symmetric=0.1):
        super().__init__()
        self.lambda_secondary = lambda_secondary
        self.lambda_mask = lambda_mask
        self.lambda_symmetric = lambda_symmetric

    def forward(self, predictions, batch):
        """
        Args:
            predictions: model output dict
            batch: ground truth dict from dataloader

        Returns:
            loss_dict: individual losses + total
        """
        loss_dict = {}

        # --- Secondary pointmap loss ---
        # Only compute on pixels where secondary GT exists
        gt_secondary = batch["world_points_secondary"]  # (B, S, H, W, 3)
        mask_secondary = batch["point_masks_secondary"]  # (B, S, H, W)
        pred_secondary = predictions["secondary_points"]  # (B, S, H, W, 3)

        if mask_secondary.sum() > 0:
            # L1 loss on valid secondary-path pixels
            diff = (pred_secondary - gt_secondary).abs()  # (B, S, H, W, 3)
            diff_masked = diff[mask_secondary]  # (N, 3)
            loss_secondary = diff_masked.mean()
        else:
            loss_secondary = torch.tensor(0.0, device=pred_secondary.device)

        loss_dict["loss_secondary"] = loss_secondary

        # --- Mask loss (binary cross-entropy) ---
        # GT mask: 1 where non-Lambertian (mirror/glass/glossy)
        gt_mask = batch["point_masks_secondary"].float()  # non-Lambertian = has secondary path
        pred_mask = predictions["mask_pred"]  # (B, S, H, W), already sigmoid'd

        # Only compute where first-surface is valid
        valid_first = batch["point_masks"]  # (B, S, H, W)
        if valid_first.sum() > 0:
            loss_mask = F.binary_cross_entropy(
                pred_mask[valid_first],
                gt_mask[valid_first],
            )
        else:
            loss_mask = torch.tensor(0.0, device=pred_mask.device)

        loss_dict["loss_mask"] = loss_mask

        # --- Symmetric loss (mirror plane constraint) ---
        # For pixels predicted as non-Lambertian with high confidence,
        # enforce that secondary_point = Reflect(first_surface_point, mirror_plane)
        # Skip for pilot (complex to batch variable-length planes)
        loss_symmetric = torch.tensor(0.0, device=pred_secondary.device)
        loss_dict["loss_symmetric"] = loss_symmetric

        # --- Total ---
        total = (
            self.lambda_secondary * loss_secondary
            + self.lambda_mask * loss_mask
            + self.lambda_symmetric * loss_symmetric
        )
        loss_dict["loss_total"] = total

        return loss_dict


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on validation set. Returns Chamfer-NL and Chamfer-L."""
    model.eval()

    chamfer_nl_list = []
    chamfer_l_list = []
    mask_acc_list = []

    for batch in val_loader:
        images = batch["images"].to(device)
        gt_first = batch["world_points"].to(device)
        gt_secondary = batch["world_points_secondary"].to(device)
        mask_first = batch["point_masks"].to(device)
        mask_secondary = batch["point_masks_secondary"].to(device)

        predictions = model(images)

        # Chamfer on non-Lambertian region (secondary-path quality)
        pred_sec = predictions["secondary_points"]
        if mask_secondary.sum() > 0:
            diff_nl = (pred_sec - gt_secondary).norm(dim=-1)  # (B, S, H, W)
            chamfer_nl = diff_nl[mask_secondary].mean().item()
            chamfer_nl_list.append(chamfer_nl)

        # Chamfer on Lambertian region (first-surface, should NOT degrade)
        # Compare VGGT's frozen first-surface output vs GT
        pred_first = predictions["world_points"]
        mask_l = mask_first & ~mask_secondary  # Lambertian only
        if mask_l.sum() > 0:
            diff_l = (pred_first - gt_first).norm(dim=-1)
            chamfer_l = diff_l[mask_l].mean().item()
            chamfer_l_list.append(chamfer_l)

        # Mask accuracy
        pred_mask_binary = predictions["mask_pred"] > 0.5
        if mask_first.sum() > 0:
            acc = (pred_mask_binary[mask_first] == mask_secondary[mask_first]).float().mean().item()
            mask_acc_list.append(acc)

    results = {
        "chamfer_nl": np.mean(chamfer_nl_list) if chamfer_nl_list else float("inf"),
        "chamfer_l": np.mean(chamfer_l_list) if chamfer_l_list else float("inf"),
        "mask_acc": np.mean(mask_acc_list) if mask_acc_list else 0.0,
    }

    model.train()
    return results


# ============================================================
# Training Loop
# ============================================================

def train(args):
    # Setup distributed if multi-GPU
    distributed = args.gpus > 1
    if distributed:
        dist.init_process_group("gloo")  # MetaX C500 uses MACA, not NCCL
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    is_main = local_rank == 0

    # Data
    train_dataset = MirrorSceneDataset(
        args.data_dir, num_views=args.num_views, img_size=518
    )
    val_dataset = MirrorSceneDataset(
        args.val_dir, num_views=args.num_views, img_size=518
    ) if args.val_dir else None

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    ) if val_dataset else None

    # Model
    model = LayeredVGGT(
        pretrained_name="facebook/VGGT-1B",
        freeze_encoder=True,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    # Loss
    criterion = LayeredPointmapLoss(
        lambda_secondary=args.lambda_secondary,
        lambda_mask=args.lambda_mask,
        lambda_symmetric=args.lambda_symmetric,
    )

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.lr * 0.01
    )

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        if is_main:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt.get("model", ckpt)
        if distributed:
            model.module.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(model_state, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    # Checkpoint dir
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Training
    if is_main:
        print(f"\n{'='*60}")
        print(f"Starting Pilot Training")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size} × {args.gpus} GPUs")
        print(f"  LR: {args.lr}")
        print(f"  Train scenes: {len(train_dataset)}")
        print(f"  Val scenes: {len(val_dataset) if val_dataset else 0}")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            # Move to device
            images = batch["images"].to(device)
            for key in ["world_points", "world_points_secondary", "point_masks",
                        "point_masks_secondary", "depths", "depths_secondary"]:
                batch[key] = batch[key].to(device)

            # Forward
            predictions = model(images)

            # Loss
            loss_dict = criterion(predictions, batch)
            loss = loss_dict["loss_total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

            # Logging
            if is_main and (step + 1) % args.log_every == 0:
                avg_loss = np.mean(epoch_losses[-args.log_every:])
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [Epoch {epoch+1}/{args.epochs}] "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"L_sec: {loss_dict['loss_secondary'].item():.4f} | "
                    f"L_mask: {loss_dict['loss_mask'].item():.4f} | "
                    f"LR: {lr_now:.2e}"
                )

        # Epoch summary
        epoch_time = time.time() - t0
        avg_epoch_loss = np.mean(epoch_losses)

        if is_main:
            print(f"\n  Epoch {epoch+1} done in {epoch_time:.0f}s | Avg loss: {avg_epoch_loss:.4f}")

        # Validation
        if val_loader and is_main and (epoch + 1) % args.eval_every == 0:
            val_model = model.module if distributed else model
            results = evaluate(val_model, val_loader, device)
            print(f"  VAL | Chamfer-NL: {results['chamfer_nl']:.4f} | "
                  f"Chamfer-L: {results['chamfer_l']:.4f} | "
                  f"Mask-Acc: {results['mask_acc']:.3f}")

            # Save results to notes
            results_path = PROJECT_ROOT / "notes" / "20-Experiments" / "Pilot-Results.md"
            _append_result(results_path, epoch + 1, avg_epoch_loss, results)

        # Checkpoint
        if is_main and (epoch + 1) % args.save_every == 0:
            state = {
                "model": (model.module if distributed else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_epoch_loss,
            }
            save_path = ckpt_dir / f"pilot_epoch{epoch+1:03d}.pt"
            torch.save(state, save_path)
            # Also save as latest
            torch.save(state, ckpt_dir / "pilot_latest.pt")
            print(f"  Saved checkpoint: {save_path}")

    # Final evaluation
    if val_loader and is_main:
        val_model = model.module if distributed else model
        results = evaluate(val_model, val_loader, device)
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"  Chamfer-NL (secondary path): {results['chamfer_nl']:.4f}")
        print(f"  Chamfer-L (first surface): {results['chamfer_l']:.4f}")
        print(f"  Mask Accuracy: {results['mask_acc']:.3f}")
        print(f"{'='*60}")

        # GATE decision
        # Compare with VGGT zero-shot baseline (run separately)
        _append_result(
            PROJECT_ROOT / "notes" / "20-Experiments" / "Pilot-Results.md",
            args.epochs, avg_epoch_loss, results, final=True
        )

    if distributed:
        dist.destroy_process_group()


def _append_result(path, epoch, loss, results, final=False):
    """Append results to Obsidian note."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "" if path.exists() else (
        "---\ntags: [experiment, pilot, results]\ncreated: 2026-06\n---\n\n"
        "# Pilot Results\n\n"
        "| Epoch | Loss | Chamfer-NL | Chamfer-L | Mask-Acc | Note |\n"
        "|-------|------|-----------|---------|---------|------|\n"
    )
    note = "**FINAL**" if final else ""
    line = f"| {epoch} | {loss:.4f} | {results['chamfer_nl']:.4f} | {results['chamfer_l']:.4f} | {results['mask_acc']:.3f} | {note} |\n"
    with open(path, "a") as f:
        if header:
            f.write(header)
        f.write(line)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Pilot training for Layered Pointmap")
    parser.add_argument("--data_dir", type=str, default="data/synthetic/train")
    parser.add_argument("--val_dir", type=str, default="data/synthetic/val")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-GPU batch size (1 scene = S views)")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Views per scene to sample")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_secondary", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=0.3)
    parser.add_argument("--lambda_symmetric", type=float, default=0.0,
                        help="Symmetric loss weight (0 for pilot)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
