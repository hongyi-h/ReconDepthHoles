"""
Pilot Sanity Check — Visualization Script

Generates diagnostic visualizations to verify the experiment is correct,
independent of loss values. Catches these failure modes:

1. DATA CHECK: Is the synthetic data physically correct?
   - RGB vs depth alignment (depth edges should match RGB edges)
   - First-surface vs secondary depth should differ in mirror regions
   - Mirror plane geometry should be consistent

2. PREDICTION CHECK: Does the model output make geometric sense?
   - Predicted secondary pointmap should be "behind" the mirror plane
   - Predicted mask should highlight mirror objects in RGB
   - Error maps should show structured (not random) patterns

3. BASELINE COMPARISON: Visual diff vs VGGT zero-shot
   - VGGT first-surface head on NL region (expected: catastrophic)
   - Our secondary head on NL region (expected: correct geometry)

Usage (on C500):
  python scripts/visualize_pilot.py \
    --checkpoint checkpoints/pilot_epoch020.pt \
    --data_dir data/synthetic/val \
    --output_dir visualizations/pilot_sanity \
    --num_scenes 5
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "third_party" / "vggt"))

from data.mirror_scene_dataset import MirrorSceneDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pilot_epoch020.pt")
    parser.add_argument("--data_dir", type=str, default="data/synthetic/val")
    parser.add_argument("--output_dir", type=str, default="visualizations/pilot_sanity")
    parser.add_argument("--num_scenes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def colorize_depth(depth, vmin=None, vmax=None):
    """Convert depth to colormap image (H, W, 3) uint8."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    valid = depth > 0
    if vmin is None:
        vmin = depth[valid].min() if valid.any() else 0
    if vmax is None:
        vmax = depth[valid].max() if valid.any() else 1

    normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    normalized[~valid] = 0  # black for invalid

    colored = (cm.viridis(normalized)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored


def colorize_error(error, mask, vmax=None):
    """Error heatmap (red = high error, blue = low)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    if vmax is None:
        vmax = error[mask].max() if mask.any() else 1
    normalized = np.clip(error / (vmax + 1e-8), 0, 1)
    colored = (cm.hot(normalized)[:, :, :3] * 255).astype(np.uint8)
    colored[~mask] = 0
    return colored


def save_grid(images, labels, path, title=""):
    """Save a grid of images with labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def run_sanity_check(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = MirrorSceneDataset(args.data_dir, num_views=8, img_size=518, max_scenes=args.num_scenes)
    print(f"Loaded {len(dataset)} scenes")

    # Load model
    from train_pilot import LayeredVGGT
    model = LayeredVGGT(pretrained_name="facebook/VGGT-1B", freeze_encoder=True).to(device)

    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_state = ckpt.get("model", ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found at {args.checkpoint}, using random init")

    model.eval()

    for scene_idx in range(len(dataset)):
        batch = dataset[scene_idx]
        scene_name = os.path.basename(batch["scene_dir"])
        print(f"\n{'='*40}")
        print(f"Scene {scene_idx}: {scene_name}")
        print(f"{'='*40}")

        # Pick first view for visualization
        view_idx = 0

        images = batch["images"].unsqueeze(0).to(device)  # (1, S, 3, H, W)
        gt_first_depth = batch["depths"][view_idx].numpy()
        gt_secondary_depth = batch["depths_secondary"][view_idx].numpy()
        gt_mask_secondary = batch["point_masks_secondary"][view_idx].numpy()
        gt_world_first = batch["world_points"][view_idx].numpy()
        gt_world_secondary = batch["world_points_secondary"][view_idx].numpy()
        rgb = (batch["images"][view_idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # ============================================================
        # CHECK 1: Data Sanity
        # ============================================================
        print("\n[CHECK 1] Data sanity...")

        # Depth range check
        d1_valid = gt_first_depth[gt_first_depth > 0]
        d2_valid = gt_secondary_depth[gt_secondary_depth > 0]
        print(f"  First depth: [{d1_valid.min():.2f}, {d1_valid.max():.2f}] m, coverage: {len(d1_valid)}/{gt_first_depth.size}")
        print(f"  Secondary depth: [{d2_valid.min():.2f}, {d2_valid.max():.2f}] m, coverage: {len(d2_valid)}/{gt_secondary_depth.size}")

        # Are they different?
        both_valid = (gt_first_depth > 0) & (gt_secondary_depth > 0)
        if both_valid.any():
            diff = np.abs(gt_first_depth[both_valid] - gt_secondary_depth[both_valid])
            print(f"  Depth difference (where both valid): mean={diff.mean():.3f}, max={diff.max():.3f}")
            if diff.mean() < 0.01:
                print(f"  ⚠️ WARNING: First and secondary depth nearly identical! Possible data bug.")
        else:
            print(f"  ⚠️ WARNING: No pixels where both depths valid.")

        # Visualize data
        depth_vmin = min(d1_valid.min(), d2_valid.min()) if len(d2_valid) > 0 else d1_valid.min()
        depth_vmax = max(d1_valid.max(), d2_valid.max()) if len(d2_valid) > 0 else d1_valid.max()

        save_grid(
            [rgb,
             colorize_depth(gt_first_depth, depth_vmin, depth_vmax),
             colorize_depth(gt_secondary_depth, depth_vmin, depth_vmax),
             (gt_mask_secondary[:, :, None] * 255).astype(np.uint8).repeat(3, axis=-1)],
            ["RGB", "First-Surface Depth", "Secondary Depth (GT)", "NL Mask (GT)"],
            output_dir / f"scene{scene_idx:02d}_01_data.png",
            title=f"CHECK 1: Data — {scene_name}"
        )

        # ============================================================
        # CHECK 2: Model Predictions
        # ============================================================
        print("\n[CHECK 2] Model predictions...")

        predictions = model(images)

        # Get predictions for first view
        pred_secondary = predictions["secondary_points"][0, view_idx].cpu().numpy()  # (H, W, 3)
        pred_mask = predictions["mask_pred"][0, view_idx].cpu().numpy()  # (H, W)
        pred_first = predictions["world_points"][0, view_idx].cpu().numpy()  # (H, W, 3)

        print(f"  Pred secondary range: [{pred_secondary.min():.2f}, {pred_secondary.max():.2f}]")
        print(f"  Pred mask range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
        print(f"  GT world_first range: [{gt_world_first[gt_world_first!=0].min():.2f}, {gt_world_first[gt_world_first!=0].max():.2f}]")

        # Compute errors
        mask_first = batch["point_masks"][view_idx].numpy()
        error_secondary = np.linalg.norm(pred_secondary - gt_world_secondary, axis=-1)  # (H, W)
        error_first = np.linalg.norm(pred_first - gt_world_first, axis=-1)

        if gt_mask_secondary.any():
            print(f"  Error on NL region (secondary head): mean={error_secondary[gt_mask_secondary].mean():.3f}")
        if mask_first.any():
            print(f"  Error on L region (first head, frozen): mean={error_first[mask_first & ~gt_mask_secondary].mean():.3f}")

        # Key check: VGGT first-surface head error on NL region
        # (This should be HIGH — it treats mirror reflections as real geometry)
        if gt_mask_secondary.any():
            vggt_error_on_nl = error_first[gt_mask_secondary].mean()
            our_error_on_nl = error_secondary[gt_mask_secondary].mean()
            print(f"\n  *** KEY COMPARISON ***")
            print(f"  VGGT first-surface on NL region: {vggt_error_on_nl:.3f}")
            print(f"  Our secondary head on NL region: {our_error_on_nl:.3f}")
            print(f"  Improvement: {(vggt_error_on_nl - our_error_on_nl) / vggt_error_on_nl * 100:.1f}%")

        # Visualize predictions
        save_grid(
            [rgb,
             colorize_error(error_first, mask_first, vmax=5.0),
             colorize_error(error_secondary, gt_mask_secondary, vmax=5.0),
             (np.clip(pred_mask, 0, 1)[:, :, None] * 255).astype(np.uint8).repeat(3, axis=-1)],
            ["RGB",
             f"First-Head Error (L+NL)",
             f"Secondary-Head Error (NL only)",
             "Predicted Mask"],
            output_dir / f"scene{scene_idx:02d}_02_predictions.png",
            title=f"CHECK 2: Predictions — {scene_name}"
        )

        # ============================================================
        # CHECK 3: Geometric Consistency
        # ============================================================
        print("\n[CHECK 3] Geometric consistency...")

        # The secondary pointmap should be "behind" the mirror plane
        mirror_planes = batch["mirror_planes"].numpy()
        if len(mirror_planes) > 0:
            plane = mirror_planes[0]  # (4,): nx, ny, nz, d
            normal = plane[:3]
            d = plane[3]

            # For valid NL pixels, check if secondary points are on the other side of mirror
            if gt_mask_secondary.any():
                # GT first-surface points in NL region
                first_pts_nl = gt_world_first[gt_mask_secondary]  # (N, 3)
                # GT secondary points in NL region
                second_pts_nl = gt_world_secondary[gt_mask_secondary]

                # Signed distance to mirror plane
                dist_first = first_pts_nl @ normal + d
                dist_second = second_pts_nl @ normal + d

                # First-surface should be on one side, secondary on the other
                same_side = np.sign(dist_first) == np.sign(dist_second)
                print(f"  Mirror plane: normal={normal}, d={d:.2f}")
                print(f"  First-surface signed dist: mean={dist_first.mean():.3f}")
                print(f"  Secondary signed dist: mean={dist_second.mean():.3f}")
                print(f"  Same side fraction: {same_side.mean():.3f} (should be LOW for correct data)")

                if same_side.mean() > 0.8:
                    print(f"  ⚠️ WARNING: Most secondary points on SAME side as first-surface!")
                    print(f"     This suggests the reflection math or raycast is wrong.")

                # Now check predictions
                pred_pts_nl = pred_secondary[gt_mask_secondary]
                dist_pred = pred_pts_nl @ normal + d
                pred_same_side = np.sign(dist_first) == np.sign(dist_pred)
                print(f"  Predicted secondary signed dist: mean={dist_pred.mean():.3f}")
                print(f"  Pred same-side as first: {pred_same_side.mean():.3f}")

        # ============================================================
        # CHECK 4: Depth-to-3D consistency
        # ============================================================
        print("\n[CHECK 4] Depth-to-3D reprojection check...")

        # If we unproject first-surface depth with camera matrices,
        # does it match the world_points GT?
        K = batch["intrinsics"][view_idx].numpy()
        ext = batch["extrinsics"][view_idx].numpy()

        H, W = gt_first_depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        x_cam = (u - cx) * gt_first_depth / fx
        y_cam = (v - cy) * gt_first_depth / fy
        z_cam = gt_first_depth
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)

        cam_to_world = np.linalg.inv(ext)
        pts_world_reproj = (cam_to_world @ pts_cam.reshape(-1, 4).T).T.reshape(H, W, 4)[:, :, :3]

        reproj_error = np.linalg.norm(pts_world_reproj - gt_world_first, axis=-1)
        valid_reproj = mask_first & (gt_first_depth > 0)
        if valid_reproj.any():
            mean_reproj = reproj_error[valid_reproj].mean()
            print(f"  Reprojection error (depth->3D vs GT world_points): {mean_reproj:.6f}")
            if mean_reproj > 0.01:
                print(f"  ⚠️ WARNING: Reprojection error > 1cm! Camera matrices or depth may be inconsistent.")
            else:
                print(f"  ✓ Consistent (< 1cm)")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nFiles to inspect:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    args = parse_args()
    run_sanity_check(args)
