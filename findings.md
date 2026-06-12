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

## 2026-06-12 — Phase 0.5 Gate 3: True Dual-Head Layered Baseline

### Verdict

**Gate 3 passed** for the in-domain synthetic layered claim.

### Evidence

Source files:

- `notes/20-Experiments/Phase-0.5-Dual-Head-Baseline.md`
- `notes/20-Experiments/phase05_dual_head_baseline.json`
- `logs/train_dual_head_baseline.log`

Setup:

- Train data: `data/synthetic/train`
- Val data: `data/synthetic/val`
- Train scenes: `1000`
- Val scenes: `200`
- Views per scene: `8`
- Epochs: `20`
- Device: `cuda`

Key metrics:

- Dual primary vs first GT on NL pixels: `0.268871`
- Dual secondary vs secondary GT on NL pixels: `0.338775`
- Dual two-layer mean on NL pixels: `0.303823`
- Dual primary vs first GT on Lambertian pixels: `0.311777`
- Dual mask accuracy: `98.565%`

Against the single-head composite baseline:

- Single-head corrected vs secondary GT on NL pixels: `0.370457`
- Dual secondary vs secondary GT on NL pixels: `0.338775`
- Dual secondary improves by `8.55%`.
- Single-head corrected vs first GT on NL pixels: `1.436062`
- Dual primary vs first GT on NL pixels: `0.268871`
- Dual primary improves first-layer recovery by `81.28%`.

### Supported Claim After Gate 3

> On in-domain synthetic mirror validation data, a true dual-head layered pointmap model can recover both first-surface and secondary-path geometry at non-Lambertian pixels. A single corrected pointmap can match secondary geometry but cannot represent both layers simultaneously.

### Still Unsupported

- Cross-domain / real-data generalization.
- Oracle-free parity against oracle plane/mask variants.
- Reflect3r comparison.
- Glass, glossy metal, wet-floor generalization.
- Multi-seed stability.

### Next Gate

Run a **cross-domain smoke test** with the dual-head checkpoint before investing in larger training. The next result should answer whether the dual-head separation survives outside the synthetic validation distribution.

## 2026-06-12 — Phase 0.5 Gate 4 Prepared: Cross-Domain Smoke

### Status

**Prepared, not yet executed.**

### Added Evaluation Entry

Source files:

- `scripts/evaluate_dual_head.py`
- `notes/20-Experiments/Phase-0.5-Dual-Head-Cross-Domain.md`

The evaluator loads `checkpoints/phase05_dual_head/dual_head_final.pt` and reports the same dual-head metrics on any `MirrorSceneDataset`-format split.

Key decision metrics:

- `dual_primary_vs_first_nl`
- `dual_secondary_vs_secondary_nl`
- `dual_two_layer_mean_nl`

The default smoke rule compares these against `notes/20-Experiments/phase05_dual_head_baseline.json` and passes only if all ratios are at most `2.0x`. It also checks NL mask coverage: predicted NL coverage must be at least `0.25x` GT NL coverage.

### Data Requirement

This gate requires a converted cross-domain split with first-surface and secondary-path GT in the current synthetic scene layout:

- `scene_XXXXX/scene_meta.json`
- `rgb_*.png`
- `depth_first_*.npy`
- `depth_secondary_*_mirror00.npy`
- `camera_*.npz`

If no real or converted split exists, use the weak synthetic fallback command recorded in [[Phase-0.5-Dual-Head-Cross-Domain]]. That fallback is lower-value than real data and must not be described as real cross-domain generalization.

### Claim Impact

No new cross-domain evidence has been produced yet. The supported claim remains:

> In-domain synthetic mirror dual-head separation works; cross-domain robustness is still unproven.
