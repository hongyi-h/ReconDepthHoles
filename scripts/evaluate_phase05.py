"""
Phase 0.5 evidence-gate evaluator.

This script computes the fair frozen-VGGT non-Lambertian baseline on the
same synthetic split and masks used by the layered pilot. Optionally pass a
layered checkpoint to report the trained secondary head on the same batches.

Primary question:
  Is the layered secondary-path prediction better than frozen VGGT on the
  same non-Lambertian pixels, same split, and same metric?
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "third_party" / "vggt"))
sys.path.insert(0, str(SCRIPT_DIR))

from data.mirror_scene_dataset import MirrorSceneDataset, collate_fn
from train_pilot import LayeredVGGT
from vggt.models.vggt import VGGT


def choose_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_layered_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    cleaned = {}
    for key, value in state.items():
        cleaned[key.removeprefix("module.")] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return {"missing": len(missing), "unexpected": len(unexpected)}


@torch.no_grad()
def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    dataset = MirrorSceneDataset(
        args.data_dir,
        num_views=args.num_views,
        img_size=args.img_size,
        split="val",
        max_scenes=args.max_scenes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    layered = None
    baseline = None
    ckpt_load = None
    if args.layered_checkpoint:
        layered = LayeredVGGT(
            pretrained_name=args.pretrained_name,
            freeze_encoder=True,
        ).to(device)
        ckpt_load = load_layered_checkpoint(layered, args.layered_checkpoint, device)
        layered.eval()
    else:
        baseline = VGGT.from_pretrained(args.pretrained_name).to(device)
        baseline.eval()

    metrics = {
        "vggt_primary_vs_first_l": MetricAccumulator(),
        "vggt_primary_vs_first_nl": MetricAccumulator(),
        "vggt_primary_vs_secondary_nl": MetricAccumulator(),
        "gt_first_vs_secondary_nl": MetricAccumulator(),
    }
    if layered is not None:
        metrics.update({
            "layered_primary_vs_first_l": MetricAccumulator(),
            "layered_primary_vs_first_nl": MetricAccumulator(),
            "layered_secondary_vs_secondary_nl": MetricAccumulator(),
            "layered_secondary_vs_first_nl": MetricAccumulator(),
            "layered_mask_acc_valid_first": MetricAccumulator(),
        })

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        gt_first = batch["world_points"].to(device)
        gt_secondary = batch["world_points_secondary"].to(device)
        mask_first = batch["point_masks"].to(device)
        mask_secondary = batch["point_masks_secondary"].to(device)
        mask_l = mask_first & ~mask_secondary
        mask_nl = mask_first & mask_secondary

        if layered is not None:
            base_pred = layered(images)
        else:
            base_pred = baseline(images)
        base_first = base_pred["world_points"]

        metrics["vggt_primary_vs_first_l"].add_l2(base_first, gt_first, mask_l)
        metrics["vggt_primary_vs_first_nl"].add_l2(base_first, gt_first, mask_nl)
        metrics["vggt_primary_vs_secondary_nl"].add_l2(base_first, gt_secondary, mask_nl)
        metrics["gt_first_vs_secondary_nl"].add_l2(gt_first, gt_secondary, mask_nl)

        if layered is not None:
            layered_first = base_pred["world_points"]
            layered_secondary = base_pred["secondary_points"]
            layered_mask = base_pred["mask_pred"] > args.mask_threshold

            metrics["layered_primary_vs_first_l"].add_l2(layered_first, gt_first, mask_l)
            metrics["layered_primary_vs_first_nl"].add_l2(layered_first, gt_first, mask_nl)
            metrics["layered_secondary_vs_secondary_nl"].add_l2(
                layered_secondary, gt_secondary, mask_nl
            )
            metrics["layered_secondary_vs_first_nl"].add_l2(
                layered_secondary, gt_first, mask_nl
            )
            metrics["layered_mask_acc_valid_first"].add_binary_acc(
                layered_mask, mask_secondary, mask_first
            )

        if args.limit_batches is not None and (batch_idx + 1) >= args.limit_batches:
            break

    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "data_dir": args.data_dir,
        "num_scenes": len(dataset),
        "num_views": args.num_views,
        "batch_size": args.batch_size,
        "device": str(device),
        "pretrained_name": args.pretrained_name,
        "layered_checkpoint": args.layered_checkpoint,
        "checkpoint_load": ckpt_load,
        "metrics": {key: acc.mean() for key, acc in metrics.items()},
        "counts": {key: acc.count for key, acc in metrics.items()},
    }

    if layered is not None:
        base = result["metrics"]["vggt_primary_vs_secondary_nl"]
        ours = result["metrics"]["layered_secondary_vs_secondary_nl"]
        if np.isfinite(base) and np.isfinite(ours) and base > 0:
            result["derived"] = {
                "secondary_nl_reduction_vs_vggt_primary_to_secondary": (base - ours) / base,
            }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.output_note:
        write_note(Path(args.output_note), result)

    print(json.dumps(result, indent=2))


def write_note(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    lines = [
        "---",
        "tags: [experiment, phase-0-5, baseline, result-to-claim]",
        f"created: {result['created']}",
        "---",
        "",
        "# Phase 0.5 Fair Baseline Evaluation",
        "",
        "## Setup",
        "",
        f"- Data: `{result['data_dir']}`",
        f"- Scenes: `{result['num_scenes']}`",
        f"- Views per scene: `{result['num_views']}`",
        f"- Device: `{result['device']}`",
        f"- Pretrained: `{result['pretrained_name']}`",
        f"- Layered checkpoint: `{result['layered_checkpoint']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Count |",
        "|---|---:|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value:.6f} | {result['counts'][key]} |")

    if "derived" in result:
        lines.extend(["", "## Derived"])
        for key, value in result["derived"].items():
            lines.append(f"- `{key}`: {value:.6f}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "This note is generated by `scripts/evaluate_phase05.py`. Treat the result as the first Phase 0.5 gate: frozen VGGT and optional layered checkpoint are evaluated on the same non-Lambertian pixels and same GT.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 0.5 fair baselines.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic/val")
    parser.add_argument("--layered_checkpoint", type=str, default=None)
    parser.add_argument("--pretrained_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, default="notes/20-Experiments/phase05_fair_baseline.json")
    parser.add_argument("--output_note", type=str, default="notes/20-Experiments/Phase-0.5-Fair-Baseline.md")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
