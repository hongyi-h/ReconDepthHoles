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

### Gate 2 First Launch Failure

`logs/train_single_head_baseline.log` showed the first Gate 2 launch failed before training started:

- Dataset discovery succeeded: `1000` train scenes and `200` val scenes.
- Model construction succeeded: `1321.8M` total params, `131.3M` trainable params.
- Failure occurred during `torch.optim.AdamW(...)` construction.
- Root cause: PyTorch Dynamo/Triton backend discovery imported the MetaX Triton backend with `MACA_HOME=None`, causing `TypeError: expected str, bytes or os.PathLike object, not NoneType`.

This is a C500 software-environment compatibility issue, not a negative experiment result.

Patch applied:

- `scripts/train_single_head_baseline.py` now sets `TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1` before importing `torch`.

Re-run the same Gate 2 command after syncing the patched script.

## Gate 2 Result

Gate 2 completed on C500:

- Single-head corrected pointmap vs secondary GT on NL pixels: `0.370457`
- Layered secondary head vs secondary GT on NL pixels: `0.407244`
- Single-head corrected pointmap vs first GT on Lambertian pixels: `0.544384`
- Single-head mask accuracy: `98.546%`

Decision:

> The current secondary-only metric does **not** prove layered representation is necessary. A single corrected pointmap trained with comparable supervision matches or beats the existing layered pilot on synthetic mirror secondary-path prediction.

Updated interpretation:

> The layered contribution must be evaluated as simultaneous recovery of first-surface and secondary-path geometry at non-Lambertian pixels, not merely corrected/secondary geometry prediction.

## Gate 3: True Dual-Head Layered Baseline

Prepared script:

- `scripts/train_dual_head_baseline.py`

Purpose:

> Train a true dual-head model with separate trainable first-surface and secondary-path heads. This is the first experiment that can support the real layered claim.

Targets:

- Primary head: first-surface GT on all valid first-surface pixels.
- Secondary head: secondary-path GT on NL pixels.
- Mask head: NL mask.

C500 command:

```bash
cd /mnt/afs/zhengmingkai/hhy/ReconDepthHoles
python scripts/train_dual_head_baseline.py \
  --data_dir data/synthetic/train \
  --val_dir data/synthetic/val \
  --num_views 8 \
  --batch_size 1 \
  --num_workers 4 \
  --epochs 20 \
  --output_json notes/20-Experiments/phase05_dual_head_baseline.json \
  --output_note notes/20-Experiments/Phase-0.5-Dual-Head-Baseline.md \
  > logs/train_dual_head_baseline.log 2>&1
```

Decision rule:

- Compare `dual_secondary_vs_secondary_nl` against single-head `0.370457`.
- Compare `dual_primary_vs_first_nl` against single-head `single_corrected_vs_first_nl = 1.436062`.
- The layered route is justified only if the dual-head model recovers both layers: low primary first-surface NL error and low secondary-path NL error.

## Gate 3 Result

Gate 3 passed on C500:

- Dual primary vs first GT on NL pixels: `0.268871`
- Dual secondary vs secondary GT on NL pixels: `0.338775`
- Dual two-layer mean on NL pixels: `0.303823`
- Dual mask accuracy: `98.565%`

Comparison against single-head:

- Single-head secondary/corrected error on NL pixels: `0.370457`
- Dual secondary error on NL pixels: `0.338775`
- Single-head first-layer error on NL pixels: `1.436062`
- Dual primary first-layer error on NL pixels: `0.268871`

Decision:

> The true dual-head model supports the narrowed layered claim on in-domain synthetic mirrors: it recovers both first-surface and secondary-path geometry at NL pixels, while the single-head baseline can only choose one layer.

Next:

> Run a cross-domain smoke test with the dual-head checkpoint before further synthetic optimization.

## Gate 4: Dual-Head Cross-Domain Smoke Test

Prepared script:

- `scripts/evaluate_dual_head.py`

Purpose:

> Test whether the trained true dual-head checkpoint still recovers both first-surface and secondary-path geometry outside the in-domain synthetic validation split.

This gate is evaluation-only. Do not retrain before it; otherwise the result no longer answers cross-domain survival of the current representation.

Required data contract:

- `scene_XXXXX/scene_meta.json`
- `scene_XXXXX/rgb_000.png ... rgb_NNN.png`
- `scene_XXXXX/depth_first_000.npy ... depth_first_NNN.npy`
- `scene_XXXXX/depth_secondary_000_mirror00.npy ... depth_secondary_NNN_mirror00.npy`
- `scene_XXXXX/camera_000.npz ... camera_NNN.npz` with `intrinsic` and `extrinsic` arrays

If the cloud repo does not contain such a converted cross-domain split, this gate is blocked by data rather than by method failure. Do not count a missing split as evidence for or against the claim.

C500 command:

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

Replace `data/synthetic/cross_domain_val` with the actual converted split path. If only a small split exists, add `--max_scenes 20` for a cheap smoke run first.

Weak synthetic fallback:

If no real or converted cross-domain split exists, generate a weak OOD synthetic split before evaluation. This is lower-value than real/converted cross-domain evidence, but it is still better than further optimizing the original synthetic validation distribution.

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

Decision rule:

- Required finite metrics:
  - `dual_primary_vs_first_nl`
  - `dual_secondary_vs_secondary_nl`
  - `dual_two_layer_mean_nl`
- Default smoke pass: all three are at most `2.0x` the in-domain reference values from `phase05_dual_head_baseline.json`.
- Smoke fail: one or more required metrics exceed `2.0x`, or predicted NL coverage is below `0.25x` GT NL coverage.
- Inconclusive: missing secondary GT, missing masks, zero NL pixels, incompatible scale, or no converted split.

Stop condition:

> If Gate 4 fails or is inconclusive, prioritize data/protocol diagnosis before additional in-domain synthetic optimization. The CVPR-level claim cannot move from "in-domain synthetic" to "robust layered reconstruction" without at least one cross-domain smoke result.
