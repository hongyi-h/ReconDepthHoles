---
tags: [experiment, phase-0-5, verification, c500, mps]
created: 2026-06-09
status: ready-for-c500
---

# Phase 0.5 Verification Start

## Decision

Start Phase 0.5 with the first evidence gate:

> Compute the fair frozen-VGGT non-Lambertian baseline on the same synthetic validation split, same masks, and same metric as the layered pilot.

This is the root check before single-head ablation, oracle comparison, or cross-domain tests.

## Local Environment Check

- Local shell Python: `/opt/homebrew/bin/python3` / `/usr/bin/python3`.
- Local default Python does not have `torch`, `numpy`, or `PIL` available.
- Local repo only has one synthetic test scene under `data/synthetic/test_batch`; the pilot validation split used by `logs/train.log` was `data/synthetic/val` with 200 scenes.
- No local pilot checkpoint was found. The training log reports the C500 checkpoint path:
  - `/mnt/afs/zhengmingkai/hhy/ReconDepthHoles/checkpoints/pilot_epoch020.pt`

Conclusion: local MPS is not currently the shortest path for the fair baseline. Use C500 for the first real verification run.

## Prepared Script

Added:

- `scripts/evaluate_phase05.py`

The script reports:

- `vggt_primary_vs_first_l`
- `vggt_primary_vs_first_nl`
- `vggt_primary_vs_secondary_nl`
- `gt_first_vs_secondary_nl`
- `layered_secondary_vs_secondary_nl` when a checkpoint is provided
- `layered_mask_acc_valid_first` when a checkpoint is provided
- derived reduction of layered secondary error vs frozen VGGT primary-to-secondary error

## C500 Command

Run on the cloud repo that contains the 200-scene validation split and pilot checkpoint:

```bash
cd /mnt/afs/zhengmingkai/hhy/ReconDepthHoles
python scripts/evaluate_phase05.py \
  --data_dir data/synthetic/val \
  --layered_checkpoint checkpoints/pilot_epoch020.pt \
  --num_views 8 \
  --batch_size 1 \
  --num_workers 4 \
  --output_json notes/20-Experiments/phase05_fair_baseline_c500.json \
  --output_note notes/20-Experiments/Phase-0.5-Fair-Baseline-C500.md
```

## Stop Condition

Do not move to Phase 1 full training until this command produces a fair baseline table.

If `layered_secondary_vs_secondary_nl` is not clearly below `vggt_primary_vs_secondary_nl`, the layered-head claim must be revised or the method debugged before adding more data.

## Gate 1 Result

Gate 1 passed on C500:

- Frozen VGGT primary vs secondary GT on NL pixels: `5.887548`
- Layered secondary head vs secondary GT on NL pixels: `0.407244`
- Relative reduction: `93.083%`

See:

- [[Phase-0.5-Fair-Baseline-C500]]
- `phase05_fair_baseline_c500.json`
- `../../logs/evaluate_phase05.log`

## Gate 2: Single-Head Composite Baseline

Prepared script:

- `scripts/train_single_head_baseline.py`

Purpose:

> Test whether a non-layered corrected pointmap head can match the layered secondary head when trained with comparable supervision.

Target definition:

- Lambertian pixels: first-surface GT
- Non-Lambertian pixels with secondary GT: secondary-path GT

C500 command:

```bash
cd /mnt/afs/zhengmingkai/hhy/ReconDepthHoles
python scripts/train_single_head_baseline.py \
  --data_dir data/synthetic/train \
  --val_dir data/synthetic/val \
  --num_views 8 \
  --batch_size 1 \
  --num_workers 4 \
  --epochs 20 \
  --output_json notes/20-Experiments/phase05_single_head_baseline.json \
  --output_note notes/20-Experiments/Phase-0.5-Single-Head-Baseline.md \
  > logs/train_single_head_baseline.log 2>&1
```

Decision rule:

- If `single_corrected_vs_secondary_nl` is close to `0.407244` and `single_corrected_vs_first_l` remains close to the frozen VGGT Lambertian error, the current evidence does not prove layered representation is necessary.
- If the single-head baseline is much worse on NL or damages Lambertian pixels, the layered path becomes materially stronger.
