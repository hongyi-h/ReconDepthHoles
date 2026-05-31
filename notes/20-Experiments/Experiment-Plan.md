---
tags: [experiment-plan, layered-pointmap, pilot, ablation, cvpr]
created: 2026-05-31
status: active
gate: pilot ≥15% non-Lambertian error reduction
---

# Experiment Plan — Layered Pointmap Reconstruction

> 从 claim 到 ablation 到 dataset 到 metric 的完整映射。每个实验都回答一个 reviewer 会问的问题。

## Claims → Experiments 映射

| # | Paper Claim | Experiment | Dataset | Metric | Pass Threshold |
|---|---|---|---|---|---|
| C1 | Layered pointmap 在非朗伯区域显著优于 VGGT/MASt3R | Main comparison table | Synth-val + Mirror3D + ScanNet++ mirror + Booster | Chamfer-NL / AbsRel-NL / δ1-NL | ≥30% Chamfer-NL 降低 vs VGGT |
| C2 | 朗伯区域不退化 | Same table, Lambertian columns | Same datasets, Lambertian mask | Chamfer-L / AbsRel-L / δ1-L | <1% 退化 |
| C3 | Zero-shot 跨域泛化（不在训练集中的真实数据） | Cross-domain eval | Mirror3D / ScanNet++ / ClearPose (全部 unseen) | Same metrics, per-dataset | 正向改进 on ≥2 datasets |
| C4 | 测试时不需要 oracle mirror plane / mask | Oracle-free ablation | Synth-val (有 GT plane) | Chamfer-NL with/without oracle | Oracle-free 版本 ≥80% of oracle 版本性能 |
| C5 | Secondary-path pointmap 恢复 metrically correct 虚像几何 | Virtual geometry eval | Synth-val (有 virtual GT) | Chamfer-Virtual / Normal-Virtual | 定性 + 定量展示 |
| C6 | 超越 Reflect3r（在其自有 setting 和我们的 setting） | Head-to-head | Reflect3r 16-scene synth + 我们的 multi-view scenes | Chamfer / AbsRel | 在 Reflect3r setting 不输；在 multi-view setting 显著胜 |

## Ablation Matrix

| ID | 变量 | 配置 A (full) | 配置 B (ablated) | 验证 claim |
|---|---|---|---|---|
| Ab1 | Layered head | First-surface + secondary-path dual head | Single head (standard VGGT) | C1: dual head 是否必要 |
| Ab2 | Symmetric loss | On | Off | C5: 对称约束是否改善 virtual geometry |
| Ab3 | Reflection mask supervision | Supervised (GT mask) | Unsupervised (no mask loss) | C4: mask 监督是否必要 |
| Ab4 | Training data | Synth mirror+glass+metal | Synth mirror-only | C1/C3: 多材质训练是否帮助泛化 |
| Ab5 | Real data augmentation | Synth + 50 real | Synth-only | C3: 真实数据是否改善跨域 |

## Phase 0: Pilot (Week 1-2) — HARD GATE

### 目标
在最小成本下验证 "layered head 能否学到 non-Lambertian 区域的几何修正"。

### 具体步骤

```
Day 1-2: 数据准备
├── 从 3DReflecNet reflective subset 下载 + 筛选 mirror/specular 类
├── 用 Blender 补充 500 mirror-scene（房间+镜子+GT virtual pointmap）
├── 总计 1k-2k 场景，split: 80% train / 10% val / 10% test
└── 生成 per-pixel: first-surface pointmap GT + secondary-path pointmap GT + material mask

Day 3-5: Model
├── Fork VGGT codebase
├── 加 secondary-path head (与 first-surface head 共享 encoder, 独立 decoder token)
├── 加 symmetric loss (mirror plane regression + reflection constraint)
├── 加 mask prediction head (binary: Lambertian / non-Lambertian)
└── Training: 2×H100, batch=4, ~24h

Day 6-8: Evaluation
├── 在 synth-val 上跑 Chamfer-NL / Chamfer-L
├── 对比 VGGT zero-shot (不 fine-tune)
├── 对比 VGGT fine-tuned on same data but single-head (Ab1 ablation)
└── 记录数字

Day 9-10: Decision
├── IF Chamfer-NL 降低 ≥15% AND Chamfer-L 退化 <2%: → PROCEED to Phase 1
├── IF 10-15%: → 调整 loss weights / 加数据 / 再跑一轮
└── IF <10%: → REJECT AND PIVOT (回到 deep-interview 重选 kernel)
```

### Pilot 输出物
- `notes/20-Experiments/Pilot-Results.md` — 数字 + 决策
- Checkpoint: `pilot-v0.1.ckpt`
- 决策: PROCEED / ITERATE / PIVOT

## Phase 1: Full Training (Week 3-8)

### 数据扩展
| Source | Type | Scenes | 用途 |
|---|---|---|---|
| 3DReflecNet reflective subset | Object-centric specular | ~5k | Pre-train / augment |
| 自建 mirror-scene (Blender) | Room + mirror + virtual GT | ~5k | 主训练 |
| 自建 glass-scene (Blender) | Room + glass panel + transmitted GT | ~2k | 泛化训练 |
| 自建 glossy-metal (Blender) | Industrial + car + glossy | ~1k | 泛化训练 |
| **Total synthetic** | | **~13k** | |

### 训练配置
- Backbone: VGGT-base (frozen encoder first 50% steps → unfreeze)
- Heads: first-surface decoder + secondary-path decoder + mask head
- Loss: L_pointmap(first) + λ1·L_pointmap(secondary) + λ2·L_symmetric + λ3·L_mask
- Hardware: 8×H100, ~3-5 days
- Hyperparameter sweep: λ1 ∈ {0.5, 1.0}, λ2 ∈ {0.1, 0.5}, λ3 ∈ {0.1, 0.3}

### 评估 (Phase 1 内部)
- Synth-val: 全指标
- Zero-shot on Mirror3D (unseen): Chamfer-NL 趋势
- 如果 zero-shot Mirror3D 不 work → 加 domain adaptation 策略

## Phase 2: Real Data + Diagnostic Protocol (Week 9-12)

### 真实数据采集 (50 场景)
```
场景类型分布:
├── 镜面 (全身镜/浴室镜/商店): 20 场景
├── 玻璃 (玻璃隔间/橱窗/玻璃桌): 15 场景
├── 抛光金属 (厨房/电梯/车身): 10 场景
└── 湿地面 (雨后/大厅): 5 场景

每场景采集:
├── Multi-view RGB (手机/DSLR, 10-30 views)
├── GT 获取: 喷哑光涂层 → 重拍 → 对齐 (AnchorD 流程)
├── Material mask: 手动标注 (Lambertian / mirror / glass / glossy)
└── Mirror plane GT (如适用): 手动标注 3 点 → 拟合平面
```

### Diagnostic Protocol 设计
```yaml
protocol:
  name: "LayeredPointmapDiag"
  version: 1.0
  
  masks:
    - lambertian: pixels where material ∈ {diffuse}
    - non_lambertian: pixels where material ∈ {mirror, glass, glossy_metal, wet}
    - secondary_visible: pixels where secondary-path GT exists
  
  metrics:
    per_region:
      - chamfer_distance (mm)
      - abs_rel
      - delta_1 (threshold 1.25)
      - normal_consistency (degrees)
    secondary_path:
      - chamfer_virtual (mm) — only on secondary_visible mask
      - normal_virtual (degrees)
    holistic:
      - PSNR / SSIM / LPIPS (NVS from reconstructed pointmap)
  
  splits:
    - in_domain: synth-test (same distribution as training)
    - cross_domain_synth: 3DReflecNet reflective (object-centric, different)
    - cross_domain_real: Mirror3D / ScanNet++ mirror / ClearPose / Booster
    - real_50: our captured real scenes
  
  baselines_required:
    - VGGT (zero-shot)
    - MASt3R (zero-shot)
    - Reflect3r (single-view mirror setting)
    - HD-VGGT (if code available)
    - Free Geometry (test-time adapted VGGT)
    - FoundationStereo (stereo setting, Booster only)
```

## Phase 3: Paper Writing (Week 13-18)

### Figure Plan
| Fig # | 内容 | 作用 |
|---|---|---|
| 1 | Teaser: VGGT 在镜面/玻璃上灾难性失效 → 我们恢复 layered geometry | Motivation |
| 2 | Method overview: shared encoder → dual decoder → symmetric loss | Architecture |
| 3 | Qualitative: 4 场景 × 3 方法 (VGGT / Reflect3r / Ours) | Visual evidence |
| 4 | Cross-domain zero-shot: bar chart on 4 unseen datasets | Killer claim |
| 5 | Virtual geometry recovery: 镜中场景的 3D 重建 | New capability |

### Table Plan
| Table # | 内容 |
|---|---|
| 1 | Main comparison: 6 baselines × 4 datasets × NL/L metrics |
| 2 | Ablation: Ab1-Ab5 on synth-val |
| 3 | Head-to-head vs Reflect3r (its setting + ours) |
| 4 | Diagnostic protocol results on Real-50 |

### Writing Timeline
- Week 13-14: Introduction + Related Work + Method
- Week 15-16: Experiments + Ablation + Figures
- Week 17: Conclusion + Supplementary
- Week 18: Internal review + polish

## Risk Registry

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Pilot fails (<10%) | 20% | CRITICAL → pivot | 先跑 pilot 再投入 |
| 3DReflecNet data 不够 mirror-scene | 30% | Medium | 自建 Blender mirror-scene 补充 |
| Real 50 采集 GT 对齐误差 >2cm | 25% | Medium | 先 dry-run 1 场景验证流程 |
| Reflect3r 代码不开源无法复现 | 15% | Low | 用 DUSt3R + virtual camera 自行复现其 pipeline |
| VGGT fine-tune 不稳定 | 20% | Medium | 先 freeze encoder 50% steps |
| Reviewer 仍判 "cosmetic vs Reflect3r" | 30% | High | Killer table: multi-view + 任意场景 + no oracle 三重差异实验 |

## 双向链接

- [[../00-Project/Project-Overview]]
- [[../10-Literature/Novelty-Check-2026-05-31]]
- [[../10-Literature/Lit-Survey-2026-05]]
- [[Pilot-Results]] — (待填)
