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

## Interpretation

This baseline predicts one composite pointmap: first-surface GT on Lambertian pixels and secondary-path GT on non-Lambertian pixels. It tests whether a non-layered corrected pointmap can match the layered secondary head.
