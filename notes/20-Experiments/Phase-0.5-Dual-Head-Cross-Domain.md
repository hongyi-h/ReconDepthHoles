---
tags: [experiment, phase-0-5, dual-head, cross-domain, ready-for-c500]
created: 2026-06-12
status: ready-for-c500
---

# Phase 0.5 Dual-Head Cross-Domain Smoke

## Status

Ready to run on C500 once a converted cross-domain split exists.

## Script

- `scripts/evaluate_dual_head.py`

## C500 Command

```bash
cd /mnt/afs/zhengmingkai/hhy/ReconDepthHoles
python scripts/evaluate_dual_head.py \
  --data_dir data/synthetic/cross_domain_val \
  --checkpoint checkpoints/phase05_dual_head/dual_head_final.pt \
  --reference_json notes/20-Experiments/phase05_dual_head_baseline.json \
  --num_views 8 \
  --batch_size 1 \
  --num_workers 4 \
  --min_nl_pixels 1000 \
  --min_pred_to_gt_nl_fraction_ratio 0.25 \
  --output_json notes/20-Experiments/phase05_dual_head_cross_domain.json \
  --output_note notes/20-Experiments/Phase-0.5-Dual-Head-Cross-Domain.md \
  > logs/evaluate_dual_head_cross_domain.log 2>&1
```

Replace `data/synthetic/cross_domain_val` with the actual converted split path.

## Weak Synthetic Fallback

If no real or converted cross-domain split exists on C500, generate a weak OOD synthetic split first:

```bash
cd /mnt/afs/zhengmingkai/hhy/ReconDepthHoles
bash scripts/batch_generate.sh \
  --num_scenes 100 \
  --num_views 8 \
  --output_dir data/synthetic/cross_domain_val \
  --jobs 8 \
  --num_mirrors 2 \
  --num_objects 9 \
  --seed_offset 100000
```

This is not a real-data generalization result. It only checks whether dual-head separation survives a distribution shift inside the current Blender generator family.

## Required Input Format

- `scene_XXXXX/scene_meta.json`
- `scene_XXXXX/rgb_000.png ... rgb_NNN.png`
- `scene_XXXXX/depth_first_000.npy ... depth_first_NNN.npy`
- `scene_XXXXX/depth_secondary_000_mirror00.npy ... depth_secondary_NNN_mirror00.npy`
- `scene_XXXXX/camera_000.npz ... camera_NNN.npz` with `intrinsic` and `extrinsic` arrays

## Decision Rule

Compare against `notes/20-Experiments/phase05_dual_head_baseline.json`.

Smoke pass requires all finite ratios below `2.0x`:

- `dual_primary_vs_first_nl`
- `dual_secondary_vs_secondary_nl`
- `dual_two_layer_mean_nl`

The evaluator also rejects a mask-collapse smoke pass if predicted NL coverage falls below `0.25x` GT NL coverage.

## Claim Impact

No cross-domain evidence has been produced yet. The current supported claim remains limited to in-domain synthetic mirrors.
