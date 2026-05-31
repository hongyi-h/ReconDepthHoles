---
tags: [project, overview, cvpr, non-lambertian, pointmap, layered-reconstruction]
created: 2026-05-31
updated: 2026-05-31
status: repositioned-post-novelty-check
ambiguity: 0.098
idea_evaluator_verdict: Strong Accept (conditional on pilot ≥15%)
---

# 非朗伯 Layered Pointmap 重建 — 项目主入口

> **One-liner (重定位后)**: Feed-forward pointmap 范式下，把场景中由 secondary light path（镜面、玻璃、抛光金属、湿地面）产生的"虚假几何"从单一 pointmap 中剥离为 **first-surface + secondary-path 多层 pointmap**，无需 test-time oracle，并以此为目标设计配套**诊断协议**。

## 关键决策（已锁定 + 重定位修正）

| 维度 | 决定 | 重定位修正 |
|---|---|---|
| 范式 | Feed-Forward Pointmap (DUSt3R/MASt3R/VGGT 系) | 不变 |
| 表征 | **Layered Pointmap** (first-surface + secondary-path heads) | ~~Real/Virtual dual pointmap~~ → Layered (0-N layers) |
| Contribution kernel | **Layered Pointmap Method + Diagnostic Protocol** | ~~"首个 non-Lambertian benchmark"~~ → 诊断协议（不拼数据量） |
| 数据策略 | 3DReflecNet reflective subset 借用 + 自建 mirror-scene 合成 + 小型真实 ~50 | 3DReflecNet 确认不撞车，可白嫖 |
| 算力 | 4-8 × H100 持续独占 4-6 周 | 不变 |
| Timeline | 质量优先，软 milestone ICCV'27 (~2027.03) | 不变 |
| 主指标 | Pointmap 几何误差（主）+ Pixel-wise depth + NVS demo | 不变 |
| 成功门槛 | **Zero-shot 跨数据集 non-Lambertian 区域 ≥30% 误差降低** | 从"仍胜"硬化为 30%+ |
| Must-have property | **测试时不需要 oracle mirror plane / mask** | 新增 |

## 撞车风险矩阵（更新后）

| 论文 | 威胁度 | 差异化策略 |
|---|---|---|
| **Reflect3r** (2509.20607) | ⚠️⚠️ | 差异 = multi-view + 任意场景 + 不需 oracle + 输出 layered pointmap（非单一 pointmap）。必须 head-to-head 实验 |
| **MirrorGaussian** (2405.11921) | ⚠️⚠️ | 差异 = feed-forward (不是 per-scene 优化) + 不限于平面镜 |
| **3DReflecNet** (2605.10204) | ⚠️ → ✅ 免费数据 | Object-centric, 无 mirror-scene, 无 virtual geometry GT。**不撞车，反而可借用 reflective subset** |
| **MS-NeRF** (2305.04268) | ⚠️ | 差异 = feed-forward (不是 per-scene NeRF) + pointmap 表征 |
| **EvalMVX** (2602.24065) | ⚠️ | 差异 = pointmap-native 诊断协议 (first-surface vs secondary-path 分别评估) |
| **GLINT** (2603.26181) | ⚠️ | 差异 = feed-forward + 不限于透明 |
| Trust3R / Free Geometry / HD-VGGT | ✅ 不冲突 | 不同坑位 (uncertainty / TTA / token suppression) |

## Idea-Evaluator 评分摘要

| Dimension | Score |
|---|---|
| Higher (accuracy gain) | 8/10 |
| Faster | 5/10 |
| Stronger (robustness/generalization) | **9/10** |
| Cheaper | 6/10 |
| Broader | 7/10 |
| Paradigm-shift probes | **4/4 Yes** |
| **Verdict** | **Strong Accept (conditional: pilot ≥15%)** |

## 工作流入口

- [[../../.omc/specs/deep-interview-non-lambertian-pointmap]] — 原始 spec（11 轮访谈）
- [[Round-Log-2026-05-30]] — Deep-interview 决策日志
- [[../10-Literature/Lit-Survey-2026-05]] — 文献调研
- [[../10-Literature/Novelty-Check-2026-05-31]] — Novelty-check 报告 + 重定位建议
- [[../20-Experiments/Experiment-Plan]] — 实验计划

## 当前阶段：Pilot → Experiment Plan

### Phase 0: Pilot (1-2 周) — GATE
- [ ] 从 3DReflecNet reflective subset 抽 1k-2k 场景
- [ ] 给 VGGT 加 layered head (first-surface + secondary-path)
- [ ] 合成验证集上跑 ablation
- [ ] **GATE**: non-Lambertian 区域 pointmap 误差 ≥15% 降低 + 朗伯不退化
- [ ] 如果 <10%: Reject and Pivot，重选 kernel

### Phase 1: Full Training (4-6 周)
- [ ] 扩展合成数据（mirror-scene + glass + glossy metal）
- [ ] Full fine-tune VGGT with layered heads
- [ ] Cross-domain zero-shot evaluation

### Phase 2: Real Data + Protocol (2-4 周)
- [ ] 50 场景真实采集 (喷哑光涂层 dry-run 先跑 1 个)
- [ ] Diagnostic protocol 设计 + 代码
- [ ] 跨域评估 (Booster / Mirror3D / ScanNet++ / ClearPose)

### Phase 3: Paper (4-6 周)
- [ ] Ablation 完整 (5 项)
- [ ] Head-to-head vs Reflect3r
- [ ] Writing + submission
