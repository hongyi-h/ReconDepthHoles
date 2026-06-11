# Findings

## 2026-06-09 — Result-to-Claim Gate: Phase 0 Pilot

### Verdict

- `claim_supported`: **no** for the intended paper claims C1-C6.
- `confidence`: **high**.
- Route: **do not proceed directly to Phase 1 full training**. Insert a Phase 0.5 evidence gate.

### What The Pilot Supports

The completed pilot supports only a narrow feasibility claim:

> On in-domain synthetic mirror data, an oracle-free layered extension of a frozen VGGT can learn a secondary-path pointmap and non-Lambertian mask.

Evidence:

- Best validation `Chamfer-NL`: `0.3956` at epoch 18.
- Final validation `Chamfer-NL`: `0.4061`.
- Final log `Chamfer-NL`: `0.4220`.
- Mask accuracy: `0.986-0.987`.
- `Chamfer-L` stays around `4.88-4.92`, consistent with the frozen first-surface head.

### What The Pilot Does Not Support

- Significant improvement over VGGT/MASt3R on non-Lambertian regions.
- Lambertian non-degradation after full fine-tuning.
- Cross-domain zero-shot generalization.
- Oracle-free performance relative to oracle plane/mask variants.
- Metrically correct virtual geometry as a full paper claim.
- Superiority or parity against Reflect3r.
- Glass, glossy metal, wet floor, or broad non-Lambertian coverage.

The previously noted `91.8%` improvement is not paper-safe because it compares the secondary head's non-Lambertian error against a frozen first-surface/Lambertian quantity, not against a fair VGGT non-Lambertian baseline on the same pixels.

### Missing Evidence

1. Frozen VGGT error on the same non-Lambertian pixels, split, and metric.
2. Single-head fine-tuned baseline to isolate whether layering helps beyond extra supervision.
3. VGGT / MASt3R / layered head-to-head on synthetic validation.
4. Oracle-free vs oracle plane/mask comparison.
5. Proper Lambertian non-degradation test with trainable model variants.
6. Zero-shot real-data tests on Mirror3D, ScanNet++ mirror, Booster, or equivalent.
7. Explicit virtual-geometry correctness evaluation against synthetic virtual GT.
8. Reflect3r comparisons in its native setting and the intended multi-view/oracle-free setting.
9. Key ablations and multi-seed variance.

### Revised Working Claim

Use only this claim until Phase 0.5 is complete:

> Preliminary results suggest layered pointmap prediction is a feasible direction for separating mirror-induced secondary geometry in feed-forward reconstruction.

Do not claim significant baseline improvement, cross-domain generalization, broad non-Lambertian handling, oracle parity, or Reflect3r superiority yet.

### Next Action: Phase 0.5 Evidence Gate

Run the smallest experiments that can decide whether the project should scale:

1. Compute the fair frozen-VGGT non-Lambertian baseline on the existing synthetic validation split.
2. Train/evaluate a single-head fine-tuned baseline on the same data.
3. Add one oracle-free vs oracle comparison.
4. Add one zero-shot real or cross-domain smoke test.
5. Re-run `result-to-claim` before Phase 1.

## 2026-06-09 — Phase 0.5 Gate 1: Fair Frozen-VGGT Baseline

### Verdict

**Gate 1 passed** on the in-domain synthetic mirror validation split.

### Evidence

Source files:

- `notes/20-Experiments/Phase-0.5-Fair-Baseline-C500.md`
- `notes/20-Experiments/phase05_fair_baseline_c500.json`
- `logs/evaluate_phase05.log`

Setup:

- Data: `data/synthetic/val`
- Scenes: `200`
- Views per scene: `8`
- Device: `cuda`
- Layered checkpoint: `checkpoints/pilot_epoch020.pt`
- Checkpoint load: `missing=0`, `unexpected=0`

Key metrics:

- Frozen VGGT primary vs secondary GT on NL pixels: `5.887548`
- Layered secondary head vs secondary GT on NL pixels: `0.407244`
- Relative reduction: `93.083%`
- Layered mask accuracy on valid first-surface pixels: `98.437%`
- Pixel count for NL metrics: `252,375,943`

### Supported Claim After Gate 1

> On in-domain synthetic mirror validation data, the trained layered secondary head predicts secondary-path geometry far more accurately than frozen VGGT's primary pointmap on the same non-Lambertian pixels.

### Still Unsupported

- Layered head vs single-head fine-tuning.
- Trainable Lambertian non-degradation.
- Oracle-free vs oracle parity.
- Cross-domain / real-data generalization.
- Reflect3r comparison.
- General glass / glossy metal / wet-floor handling.

### Next Gate

Run the **single-head fine-tuned baseline** on the same synthetic train/val split. This is now the most important confound: the current result proves that the added secondary head beats frozen VGGT, but not that a layered representation is necessary versus a non-layered model trained with comparable supervision.

## 2026-06-11 — Phase 0.5 Gate 2: Single-Head Composite Baseline

### Verdict

**Gate 2 does not support layered-necessity under the current secondary-only metric.**

### Evidence

Source files:

- `notes/20-Experiments/Phase-0.5-Single-Head-Baseline.md`
- `notes/20-Experiments/phase05_single_head_baseline.json`
- `logs/train_single_head_baseline.log`

Setup:

- Train data: `data/synthetic/train`
- Val data: `data/synthetic/val`
- Train scenes: `1000`
- Val scenes: `200`
- Views per scene: `8`
- Epochs: `20`
- Device: `cuda`

Key comparison:

- Layered secondary head vs secondary GT on NL pixels: `0.407244`
- Single-head corrected pointmap vs secondary GT on NL pixels: `0.370457`
- Single-head relative reduction vs frozen VGGT primary-to-secondary: `93.705%`
- Single-head mask accuracy: `98.546%`
- Single-head Lambertian first-surface error: `0.544384`

### Interpretation

The single-head composite baseline matches or beats the current layered pilot for the secondary-path metric. Therefore, the existing evidence does not justify claiming that layered representation is necessary for correcting synthetic mirror-region geometry.

This does **not** kill the project. It changes the required claim:

> The layered contribution must be about simultaneous recovery of first-surface and secondary-path geometry, not merely corrected secondary geometry on non-Lambertian pixels.

### Next Gate

Train and evaluate a **true dual-head model**:

1. Primary trainable head predicts first-surface GT on all valid pixels.
2. Secondary trainable head predicts secondary-path GT on NL pixels.
3. Mask head predicts NL mask.
4. Compare against the single-head baseline on a two-layer metric:
   - primary/first-surface NL error,
   - secondary-path NL error,
   - Lambertian first-surface error,
   - combined two-layer score.

If the dual-head model cannot beat the single-head baseline on the two-layer objective, the method should pivot toward "material-aware corrected pointmap" rather than "layered pointmap."
