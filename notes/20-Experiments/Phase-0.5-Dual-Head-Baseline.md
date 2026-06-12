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

## Gate Decision

**Gate 3 passed for the in-domain synthetic layered claim.**

The true dual-head model supports the revised layered contribution because it recovers both layers at non-Lambertian pixels:

- Dual primary head vs first-surface GT on NL pixels: `0.268871`
- Dual secondary head vs secondary-path GT on NL pixels: `0.338775`
- Dual two-layer mean on NL pixels: `0.303823`
- Dual mask accuracy: `98.565%`

Against the single-head composite baseline:

- Single-head corrected vs secondary GT on NL pixels: `0.370457`
- Dual secondary vs secondary GT on NL pixels: `0.338775`
- Dual secondary improves over single-head secondary by `8.55%`.
- Single-head corrected vs first GT on NL pixels: `1.436062`
- Dual primary vs first GT on NL pixels: `0.268871`
- Dual primary improves over single-head first-layer recovery by `81.28%`.

Against frozen VGGT:

- Frozen VGGT primary vs first GT on NL pixels: `5.634895`
- Frozen VGGT primary vs secondary GT on NL pixels: `5.885385`
- Dual-head reduces both errors by more than an order of magnitude on this synthetic split.

## Supported Claim

This result supports the narrowed claim:

> On in-domain synthetic mirror data, a true dual-head layered pointmap model can simultaneously recover first-surface and secondary-path geometry at non-Lambertian pixels, while a single corrected pointmap must choose one layer.

## Remaining Limits

This still does not prove:

- cross-domain or real-data generalization,
- oracle-free parity against oracle plane/mask variants,
- Reflect3r superiority,
- generalization beyond synthetic mirrors to glass, glossy metal, or wet floors,
- multi-seed stability.

## Next Gate

The next highest-value gate is a **cross-domain smoke test** with the dual-head checkpoint. Do not continue optimizing synthetic in-domain performance until one unseen-domain result exists.

## Interpretation

This baseline trains separate primary and secondary pointmap heads. It is the first Phase 0.5 test that can support a true layered claim: simultaneous first-surface and secondary-path recovery at non-Lambertian pixels.
