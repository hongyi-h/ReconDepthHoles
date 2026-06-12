---
tags: [experiment, phase-0-5, dual-head, layered]
created: 2026-06-12T12:31:31
---

# Phase 0.5 Dual-Head Layered Baseline

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
| `vggt_primary_vs_first_l` | 4.613591 | 60025237 |
| `vggt_primary_vs_first_nl` | 5.634895 | 252186954 |
| `vggt_primary_vs_secondary_nl` | 5.885385 | 252186954 |
| `dual_primary_vs_first_l` | 0.311777 | 60025237 |
| `dual_primary_vs_first_nl` | 0.268871 | 252186954 |
| `dual_secondary_vs_secondary_nl` | 0.338775 | 252186954 |
| `dual_secondary_vs_first_nl` | 1.470180 | 252186954 |
| `dual_mask_acc_valid_first` | 0.985652 | 312212191 |
| `dual_two_layer_mean_nl` | 0.303823 | 252186954 |

## Derived
- `dual_secondary_reduction_vs_vggt_primary_to_secondary`: 0.942438

## Interpretation

This baseline trains separate primary and secondary pointmap heads. It is the first Phase 0.5 test that can support a true layered claim: simultaneous first-surface and secondary-path recovery at non-Lambertian pixels.
