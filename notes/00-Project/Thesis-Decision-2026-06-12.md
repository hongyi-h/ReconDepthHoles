---
tags: [project, decision, thesis, scope, non-lambertian, layered-pointmap]
created: 2026-06-12
status: locked
supersedes: one-liner-in-Project-Overview (Thesis-A framing)
---

# Thesis 决策与 v1 Scope 锁定 (2026-06-12)

## 触发

Gate 2 证明 single-head corrected pointmap 在 NL 像素打平/胜过 layered secondary。
这逼问：**镜中 secondary 这层几何到底为什么需要？** 逐光路推演后做出 thesis 选择。

## 决策 1：选 Thesis B (Capability)，放弃 Thesis A (Robustness)

| | Thesis A: Robustness | **Thesis B: Capability (选定)** |
|---|---|---|
| 主张 | 让前向重建不被非朗伯表面骗 | 镜面反射 = 免费的第二视角，同时恢复 ①直接 first-surface ②镜面揭示的隐藏真实几何 ③自标定镜面 |
| 要 secondary 层吗 | 不要 (Gate 2 已证) | **要，不可替代** |
| 新颖性 | 弱，接近 Mirror3D | 强 |

**Why**: 对"修复深度失效"这个原始问题，detection + first-surface inpainting 就够 (Gate 2)，secondary 层多余。secondary 层唯一不可替代的价值是 capability：
1. **揭示隐藏几何** — 镜中可见相机直接视角看不到的物体背面/遮挡区，反射回真实空间 = 凭空多恢复一块真实几何 (catadioptric 原理)。
2. **无 oracle 自标定镜面** — first 点与 secondary 点关于镜面镜像对称，两层间的镜像约束反推镜面方程，兑现 one-liner 的"测试时不需要 oracle"。

## 决策 2：表征 surface-agnostic，平面性只在"反演"里

核心洞察：**平面假设不在表征里，在反演里。**
- dual-head 预测 secondary pointmap 本身不假设平面，弯镜/波纹玻璃也能逐像素回归虚拟点。
- 平面假设只在第二步进入：把 secondary 反射回真实空间(揭示隐藏几何) / 自标定镜面时，需要光路反演模型。
- 反演模型由**非朗伯界面的逐像素法向场 (+折射时 IOR)** 决定。平面 = 法向常数的特例。

→ 方法 = **自由预测 secondary 层 (surface-agnostic) + 可替换的光路反演模块**。平面是最 robust 锚点，曲面是自然推广，波纹折射是失效边界。这同时拉开与 Reflect3r 差距 (镜面工作多默认平面)。

## 决策 3：v1 可定量战线 = 平面镜锚点 + 曲面镜推广 demo (Scope 选项 1)

payoff (隐藏几何恢复) 的适定性随界面复杂度单调衰减：

| 界面 | 反演 | v1 claim 等级 |
|---|---|---|
| 平面镜 | 闭式、超定 → 干净可定量 | **主战场：定量** |
| 光滑曲面镜 | 需平滑/可积先验，误差累积 | **推广 demo (哈哈镜恢复背面)** |
| 波纹玻璃/强折射 | caustics/多对一/局部不可逆 | **future work：检测可达，几何恢复不可达** |

分级 claim，不过度声称。"我知道边界在哪"作为卖点而非弱点 → 画一条"性能 vs 界面复杂度"曲线作为 CVPR 图。

## 待补：缺失的干净 baseline (方法论根因)

Gate 3 的"dual primary 比 single-head first-layer 好 81%"是**同义反复**，不能作分层价值证据 (single-head 在 NL 被监督拟合 secondary GT，拿去和 first GT 比当然差)。

**必须补**：一个 **first-surface + mask multitask baseline (NL 像素也监督 first GT)**。
- 若 dual-head 的 first-surface 精度不优于它 → secondary 层对 first 任务零贡献 → robustness 故事彻底不成立，只剩 Thesis B (capability + 自标定) 撑场。这与选 B 一致，但需明确写出来。

## 数据策略推论 (闭环)

- secondary 层定量 GT **只有合成能给** (真实数据集无人标镜中虚拟几何)。见 [[Project-Overview]] 数据矩阵。
- 正确合成策略 = 生成**界面复杂度梯度数据族**：平面镜 → 曲面镜 → 波纹玻璃，每个带 first/secondary 两层 GT + 界面法向场 GT。Blender 可渲染曲面反射材质与 glass shader，并导出已知法向场。
- 真实数据 (Mirror3D/Booster) 只用于：① first-surface 不退化 ② NL mask 检测精度 ③ secondary 层定性证据 (NVS/反射一致性)。

## 下一步 (按依赖排序)

1. **Gate 4 解锁**：C500 生成 weak OOD synthetic split 跑 cross-domain smoke (零下载)。见 [[../20-Experiments/Phase-0.5-Dual-Head-Cross-Domain]]。
2. **补干净 baseline**：first-surface + mask multitask，与 dual-head 对比 first-surface 精度。
3. **生成器扩展**：加曲面镜材质 + 法向场 GT 导出 (波纹玻璃留 future work，但生成器预留接口)。
4. 真实数据下载 (Mirror3D 镜面子集 + Booster test 子集) 并行启动，服务 Phase 1/2。

## 关联

- [[Project-Overview]] — one-liner 需从 Thesis-A 表述更新为 B
- [[Round-Log-2026-05-30]] — 早期 deep-interview 决策
- [[../../findings.md]] — Gate 1-4 证据链
