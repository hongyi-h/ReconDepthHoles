---
tags: [novelty-check, codex-review, adversarial, decision]
created: 2026-05-31
reviewer: Codex GPT-5.5 xhigh
verdict: PROCEED_WITH_CAUTION
---

# Novelty Check Report — Non-Lambertian Pointmap (CVPR'27 候选)

> **Verdict: PROCEED_WITH_CAUTION** — 当前 framing 极可能被拒。原 spec 的 [A3] 形态需要重定位才能投。

## Proposed Method (1-line)

Feed-forward pointmap 重建模型上的 Real/Virtual 双 pointmap + symmetric loss + reflection mask + non-Lambertian benchmark。

## Codex 给的 Novelty 分数

| Contribution | Score (现状) | Score (重定位后) | 关键阻力 |
|---|---|---|---|
| Method (Real/Virtual dual pointmap + symmetric loss) | **3/10** | **6/10** | Reflect3r 已实质性占领 |
| Benchmark (10k synth + 50 real + 跨域协议) | **4/10** | **6/10** | EvalMVX + 3DReflecNet 已存在 |

## 三个高威胁但本人之前没识别到的撞车

### 1. **3DReflecNet (CVPR 2026, arxiv:2605.10204)** ⚠️⚠️⚠️ 最危险
- **22 TB**, 12k+ 形状, 120k+ 合成实例, 1k+ 真实物体, 7M+ 多视图帧
- 显式针对 reflective / transparent / low-texture 三类材质
- 已经设计了 5 类任务的 benchmark：image matching / SfM / NVS / reflection removal / relighting
- **直接打死** "首个非朗伯 benchmark" 的卖点。10k 合成 + 50 真实 在它面前完全是 subset

### 2. **MirrorGaussian (ECCV-era 2024, arxiv:2405.11921)** ⚠️⚠️
- 与原计划 [A] 概念上等价：明确做"real-world space + 镜像 counterpart reflected about mirror plane"
- 用 dual-rendering 同时光栅化 real & mirrored Gaussians, 端到端联合优化镜面平面
- 区别在 3DGS 路线（per-scene 优化）而非 feed-forward pointmap
- **撞 motivation**: Reviewer 会说"你只是把 MirrorGaussian 的 idea 搬到 pointmap"

### 3. **MS-NeRF (TPAMI 2025, arxiv:2305.04268)** ⚠️
- "Multi-space NeRF": 并行子空间表征反射/折射
- 做过 33 合成 + 7 真实 复杂反射/折射 benchmark
- 与"虚像也是另一片几何"的概念哲学一致
- **撞概念**: Reviewer 会说"早在 2023 就提出多空间表征处理反射"

### 4. **GLINT (arxiv:2603.26181)** ⚠️
- 3DGS 透明场景，分解 primary interface + reflected + transmitted radiance
- 验证了"分解式表征处理非朗伯"已是成熟方向

## Reflect3r 撞车深度分析

Codex 直接判定我之前的差异化论证（"defect vs auxiliary view"）**是 cosmetic 的**，不构成科学差异。

| 维度 | Reflect3r | 原 [A3] | 真有效差异 |
|---|---|---|---|
| 范式 | feed-forward pointmap | feed-forward pointmap | ✗ 同 |
| 利用反射 | 作为 stereo cue | 作为分解几何 | ⚠ Codex 判 cosmetic |
| 对称损失 | 已有 | 已有 | ✗ 同 |
| 输入 | 单视图 + 必须含镜子 | 多视图 + 任意场景 | ✓ 真差异 |
| 输出 | 单一 pointmap | Real + Virtual + mask 双 pointmap | ✓ 真差异 |
| 测试时是否需要镜面平面 | 隐含需要 | 不需要 (must-have) | ✓ 真差异 |
| 范围 | 仅平面镜 | 期望覆盖玻璃/抛光金属/部分透明 | 真差异，但要做出来 |

**Codex 提示**: 必须把"多视图 + 不需要 oracle mirror plane + 同时输出 metrically registered first/secondary geometry"作为差异核心，并实验上展示 Reflect3r 在 multi-view + 任意场景 setting 下败下阵。

## 重定位建议

### ❌ 不要再喊
- "首个 non-Lambertian benchmark for pointmap reconstruction"
- "Mirror as defect not as cue"
- "Real/Virtual dual-pointmap" 作为最高 contribution

### ✅ 改喊（更小、更硬、更 undeniable）

**新 one-liner**: *"A diagnostic benchmark and feed-forward model for **layered pointmap reconstruction under secondary light paths**."*

**Method 重定位为：分层（layered）pointmap 重建**
- 不仅做镜面，做"任何由次级光路产生的虚假几何"——镜面、玻璃透射、湿地面镜像、抛光金属镜像
- 一次前传输出：first-surface pointmap + secondary-path pointmap (可以是 0-N 个 layer)
- **关键 must-have**: 测试时不需要 oracle mirror plane / mask
- 与 SeeGroup (multi-layer transparent depth) 类比但运行在 pointmap 表征上

**Benchmark 重定位为：诊断协议（不是 dataset 体量竞争）**
- **抛弃**"我们造了多大 dataset"的 framing — 体量永远拼不过 3DReflecNet
- **改为**"我们提供 pointmap 范式专属的诊断协议"：per-pixel real/virtual GT、material masks、first-surface vs reflected-surface 分别评估、cross-dataset zero-shot splits
- 重点是**评估协议本身的价值**而非数据规模
- 真实 50 场景仍然有用：作为 protocol 的载体

### Killer Experimental Claim (一定要做出来)

> "Trained ONLY on our synthetic decomposition data, our model **zero-shot reduces non-Lambertian-region pointmap error by 30-50%** over VGGT / MASt3R / HD-VGGT / Free Geometry / Reflect3r on UNSEEN real datasets, while preserving diffuse-region accuracy, AND recovers metrically correct hidden mirror-visible geometry."

如果这个数字打不出来，论文会被拒。

### 必须有的 Ablation
- (a) Real/Virtual 双头 vs 单头
- (b) Symmetric loss on/off
- (c) Reflection mask 监督 on/off
- (d) Synth-only vs Synth + 50-real
- (e) **Reflect3r 在 multi-view + 任意场景下的 head-to-head**——直接在 Reflect3r 自己的 setting 也跑一下

## 最坏审稿 (Codex 演练)

> "本文与 Reflect3r 的差异主要是 cosmetic 的——把虚视图 idea 移到 pointmap 头。real/virtual 多空间表征早在 MS-NeRF (TPAMI'25)、MirrorGaussian (ECCV'24)、Mirror-3DGS、GLINT 中存在。Benchmark 部分体量、覆盖、任务多样性都不及 EvalMVX (CVPR'26 era) 和 3DReflecNet (CVPR'26)。'首个 pointmap 范式 benchmark' 是 implementation-category distinction，不是 scientific contribution。Cross-domain 提升仅证明 in-domain 训练效果，不证明能力。"

## 我推荐的下一步

不要急着进 experiment-plan。先做两件事：

1. **细读 3DReflecNet (2605.10204) 全文** — 弄清 1) 它的 GT 形态是不是 pointmap 友好的；2) 它的 split 是否真的 cover 了 mirror 反射场景里"虚像几何"；3) 它的 evaluation 是不是 first-surface only。**如果 3DReflecNet 的数据本身可以被本工作的协议直接消费，那就不是撞车而是免费 baseline 数据**——可大幅省去自建合成的 6-8 周。

2. **让 idea-evaluator skill 评估重定位后的版本** — 看新 framing 在五维框架（Higher/Faster/Stronger/Cheaper/Broader）+ paradigm-shift 下能拿几分

之后再决定是 (a) 重新跑 deep-interview 锁定新 framing 还是 (b) 直接 experiment-plan。

## 双向链接

- [[../00-Project/Project-Overview]]
- [[Lit-Survey-2026-05]]
- [[../../.omc/specs/deep-interview-non-lambertian-pointmap]] — 该 spec 的 [A3] framing 已部分被 Codex 否定，需重订
