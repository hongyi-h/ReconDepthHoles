---
tags: [experiment, phase-0-5, single-head, baseline]
created: 2026-06-11T01:47:46
---

# Phase 0.5 Single-Head Baseline

## Setup

- Train data: `data/synthetic/train`
- Val data: `data/synthetic/val`
- Train scenes: `1000`
- Val scenes: `200`
- Views per scene: `8`
- Epochs: `20`
- Device: `cuda`

## Metrics

| Metric | Value | Count |
|---|---:|---:|
| `vggt_primary_vs_first_l` | 4.614955 | 59860596 |
| `vggt_primary_vs_secondary_nl` | 5.884771 | 252326724 |
| `single_corrected_vs_first_l` | 0.544384 | 59860596 |
| `single_corrected_vs_secondary_nl` | 0.370457 | 252326724 |
| `single_corrected_vs_first_nl` | 1.436062 | 252326724 |
| `single_mask_acc_valid_first` | 0.985456 | 312187320 |

## Derived
- `single_head_reduction_vs_vggt_primary_to_secondary`: 0.937048

## Gate Decision

**Gate 2 fails to support layered-necessity under the current metric.**

The single-head composite baseline is not worse than the existing layered pilot on secondary-path prediction:

- Layered secondary head vs secondary GT on NL pixels: `0.407244`
- Single-head corrected pointmap vs secondary GT on NL pixels: `0.370457`
- Single-head relative reduction vs frozen VGGT primary-to-secondary: `93.705%`
- Single-head mask accuracy: `98.546%`

It also improves Lambertian first-surface prediction under the composite target:

- Frozen VGGT primary vs first GT on Lambertian pixels: `4.614955`
- Single-head corrected pointmap vs first GT on Lambertian pixels: `0.544384`

Therefore, the current evidence does **not** prove that a layered representation is necessary if the task is framed as "predict corrected/secondary geometry on non-Lambertian pixels."

## Critical Interpretation

This is a confound, not a project-ending failure.

The single-head baseline predicts one composite surface:

- first-surface geometry on Lambertian pixels,
- secondary-path geometry on non-Lambertian pixels.

It cannot output both first-surface and secondary-path geometry at the same non-Lambertian pixel. The current layered pilot also does not yet solve that fully, because its primary branch is frozen and remains poor on NL first-surface geometry.

The defensible next claim must become:

> A true dual-layer model can recover both first-surface and secondary-path geometry at non-Lambertian pixels, while a single-head corrected pointmap must choose one layer.

## Next Gate

Train a true dual-head model:

- primary head trained against first-surface GT on all valid pixels,
- secondary head trained against secondary-path GT on NL pixels,
- mask head trained as before.

Then compare against the single-head baseline using a two-layer metric:

- first-surface NL error,
- secondary-path NL error,
- combined two-layer score,
- Lambertian first-surface error.

## Interpretation

This baseline predicts one composite pointmap: first-surface GT on Lambertian pixels and secondary-path GT on non-Lambertian pixels. It tests whether a non-layered corrected pointmap can match the layered secondary head.
