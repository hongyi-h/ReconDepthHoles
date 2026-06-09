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

