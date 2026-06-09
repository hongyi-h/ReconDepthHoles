---
tags: [literature, survey, non-lambertian, pointmap, mirror-reconstruction, feed-forward-3D]
created: 2026-06-09
date_window: "2024-01 → 2026-06"
---

# 文献调研 — 非朗伯表面 Pointmap 重建 (2024-2026.06)

> 全新独立调研，覆盖 feed-forward 3D 重建、非朗伯/镜面/透明深度估计、3DGS 镜面处理四大方向。

---

## 一、Feed-Forward 3D 重建基础模型（Pointmap 路线）

这是本项目的技术基座。核心趋势：从 per-scene 优化转向单次前传预测 dense 3D geometry。

| 论文 | 时间 | Venue | 核心贡献 | 与本项目关系 |
|------|------|-------|---------|-------------|
| **DUSt3R** | 2024.03 | CVPR'24 | 双视图前传 pointmap 预测，无需标定 | 开山之作，MASt3R/VGGT 基线 |
| **MASt3R** | 2024.06 | CVPR'24 | DUSt3R + 局部特征匹配，提升精度 | 直接 baseline |
| **VGGT** (Visual Geometry Grounded Transformer) | 2025.03 | CVPR'25 | 1.2B 参数，单前传同时输出 camera/depth/pointmap/track | **本项目的骨干模型** |
| **G3T Up!** [2605.27372] | 2026.05 | arXiv | 重力对齐坐标系简化 pointmap 后处理 | VGGT 后续改进，坐标系选择 insight |
| **Trust3R** [2605.19539] | 2026.05 | arXiv | Evidential uncertainty (NIW 分布) 为 pointmap 加置信度 | **占位**：uncertainty 路线已有人做 |
| **Free Geometry** [2604.14048] | 2026.04 | arXiv | Test-time self-refinement via cross-view consistency LoRA | **占位**：TTA 路线已有人做 |
| **QVGGT** [2605.31124] | 2026.05 | arXiv | VGGT 后训练量化 (INT4/INT8) | 部署优化，与本项目正交 |
| **VG²GT** [2606.01573] | 2026.06 | arXiv | Voxel-Gaussian + VGGT，feed-forward 3DGS | 与 VGGT 结合的新表征 |
| **IVGT** [2605.16258] | 2026.05 | arXiv | Implicit 版本 VGGT，NeRF-style 连续表征 | 与 pointmap 互补的新方向 |
| **R³** [2605.26519] | 2026.05 | arXiv | 相对回归避免全局坐标系假设 | 解耦坐标系 |
| **FF3R** [2604.09862] | 2026.04 | arXiv | Feed-forward 3D + 语义特征联合重建 | 多任务前传 |
| **Anchor3R** [2606.05035] | 2026.06 | arXiv | Streaming 3D with transient anchors，长序列在线建图 | 长序列扩展 |
| **ScaRF-SLAM** [2606.00307] | 2026.05 | arXiv | GFM + classical SLAM 统一 | SLAM 集成 |
| **ViGeo** [2605.30060] | 2026.05 | arXiv | 视频级时空一致几何恢复 | 时序扩展 |
| **Robust 4D VGGT** [2604.09366] | 2026.04 | arXiv | 动态 4D 场景下加 uncertainty prior | 动态场景扩展 |
| **CoMo3R-SLAM** [2605.30488] | 2026.05 | arXiv | 多智能体协作 dense SLAM + pointmap prior | 多机协作 |

**关键观察**：
1. VGGT 生态正在快速膨胀（量化、4D、SLAM、语义），但 **没有任何工作针对非朗伯失效**
2. Trust3R 用 uncertainty 来标记"不可信区域"但不修复它们
3. Free Geometry 做 TTA 但假设所有区域的 cross-view consistency 规则相同——镜面/玻璃天然违反这个假设

---

## 二、非朗伯/镜面/反射表面 3D 重建

本项目的核心问题域。分为 per-scene 优化（3DGS/NeRF）和 feed-forward 两个流派。

### 2.1 Per-Scene 优化路线（3DGS/NeRF）

| 论文 | 时间 | 核心方法 | 局限 |
|------|------|---------|------|
| **MirrorGaussian** [2405.11921] | 2024.05 | 检测镜面 → 虚拟相机 → dual-branch 3DGS | 需要 oracle mirror mask，per-scene 优化 |
| **Mirror-3DGS** [2404.01168] | 2024.04 | 镜面反射建模 + 3DGS | 需要 mirror plane 先验 |
| **GS in Mirrors** [2410.01614] | 2024.10 | 虚拟相机优化处理镜面 3DGS | 仅处理平面镜 |
| **Seeing Through Reflections (SSR-GS)** [2509.18956] | 2025.09 | GS + 反射分离 | Per-scene, 仅平面镜 |
| **TR-Gaussians** [2511.13009] | 2025.11 | 透射+反射 planar 分层 3DGS | 仅处理平面玻璃 |
| **NeRFs are Mirror Detectors** [2501.04074] | 2025.01 | SSIM 检测镜面 + surface primitives | NeRF 路线，慢 |

**关键观察**：
- 所有 3DGS/NeRF 镜面方法都是 **per-scene 优化**（10-30 min/scene）
- 多数需要 **oracle 输入**（mirror mask 或 mirror plane）
- 仅处理 **平面镜**，不覆盖曲面金属、抛光地面、玻璃

### 2.2 Feed-Forward / 单前传路线

| 论文 | 时间 | 核心方法 | 局限 |
|------|------|---------|------|
| **Reflect3r** [2509.20607] | 2025.09 | 镜面反射=免费 stereo pair → DUSt3R 变体重建 | **主要竞品**。单视图+镜面限定；需要包含镜子的图像对；不处理非镜面反射 |
| **EndoVGGT** [2603.24577] | 2026.03 | GNN 增强 VGGT 解决内窥镜 specular | 医学领域限定，不是通用方法 |

**关键观察**：
- Reflect3r 是唯一在 feed-forward 路线做镜面的工作，但它的 setting 非常窄：**必须图中可见镜面反射**，本质是利用镜像构造 stereo pair
- **没有任何工作**在通用 feed-forward pointmap 模型上解决非朗伯失效（镜面、玻璃、金属、湿地面的统一处理）

---

## 三、透明/反射物体深度估计

| 论文 | 时间 | 核心方法 | 与本项目关系 |
|------|------|---------|-------------|
| **SeeGroup** [2605.28735] | 2026.05 | Self-determined grouping 多层深度（透明表面+背后物体） | **高度相关**：多层深度思路类似，但限于透明物体 + 单目 |
| **Trans2Occ** [2606.01777] | 2026.06 | Sim2Real 透明物体 voxel occupancy + grasp | 合成→真实迁移思路可参考 |
| **SeeClear** [2603.19547] | 2026.03 | Generative opacification：先"不透明化"再估计深度 | 有趣思路但需要 image generation |
| **TransNormal** [2602.00839] | 2026.01 | Dense visual semantics → diffusion-based 透明物体法线 | 法线估计，单任务 |
| **DKT (Diffusion Knows Transparency)** [2512.23705] | 2025.12 | Video diffusion repurposed for transparent depth+normal | ~11k 合成视频数据，benchmark 参考 |
| **AnchorD** [2605.02667] | 2026.05 | Factor-graph grounding 单目深度 → metric (透明/反射/非朗伯) | **重要参考**：测试时融合多传感器，不改模型 |
| **CDPR** [2604.11097] | 2026.04 | Cross-modal diffusion + 偏振 → 单目深度 | 需要偏振硬件 |
| **Depth Ambiguity (Flying Points)** [2606.02552] | 2026.06 | Mixture density 表征深度歧义 → 消除 flying points | 相关：深度歧义的概率建模 |
| **EGSA-PT** [2511.14970] | 2025.11 | Edge-guided spatial attention 透明物体深度+分割 | 多任务 baseline |

**关键观察**：
- **SeeGroup** 最接近我们的"分层"思路，但限于透明物体+单目，不是 pointmap 路线
- AnchorD 用后处理修复非朗伯深度，但 **不改模型本身**
- 没有任何工作在 feed-forward pointmap 模型内部做非朗伯分层

---

## 四、其他相关辅助方法

| 论文 | 时间 | 关系 |
|------|------|------|
| **Glare-Resilient Navigation Costmap** [2604.12753] | 2026.04 | Specular glare 对 depth 的破坏 + 融合修复（下游应用视角） |
| **Shoot-Bounce-3D** [2512.06080] | 2025.12 | LiDAR 二次反射分解遮挡+镜面 | 物理模态不同但"分解光路"概念类似 |
| **C3VD-DEFCOL** [2606.07891] | 2026.06 | 带 specular 的肠镜 3D 数据集 | 有 GT，但医学专用 |
| **MODEST** [2511.20853] | 2025.11 | 多光学参数 stereo dataset | 评估方法论参考 |

---

## 五、Landscape 综合分析

### 5.1 整体格局

```
                    Per-Scene 优化              Feed-Forward
                 ┌──────────────────┐    ┌──────────────────────┐
  朗伯表面       │ 3DGS/NeRF 成熟   │    │ VGGT/MASt3R/DUSt3R   │
  (正常)         │                  │    │ 生态爆发             │
                 └──────────────────┘    └──────────────────────┘
                 ┌──────────────────┐    ┌──────────────────────┐
  镜面           │ MirrorGaussian    │    │ Reflect3r            │
  (平面镜限定)   │ Mirror-3DGS       │    │ (单视图+镜像stereo)  │
                 │ SSR-GS, TR-GS    │    │                      │
                 └──────────────────┘    └──────────────────────┘
                 ┌──────────────────┐    ┌──────────────────────┐
  透明物体       │ (少量 NeRF 工作)  │    │ SeeClear, SeeGroup   │
                 │                  │    │ DKT, TransNormal      │
                 └──────────────────┘    └──────────────────────┘
                 ┌──────────────────┐    ┌──────────────────────┐
  通用非朗伯     │                  │    │                      │
  (镜面+玻璃+    │    ← 稀疏 →      │    │    ← 完全空白 →      │
   金属+湿地面)  │                  │    │                      │
                 └──────────────────┘    └──────────────────────┘
```

### 5.2 已确认的真空白

1. **Feed-forward pointmap × 非朗伯几何分解** — 完全空白
2. **Pointmap 内部多层/分层预测（first-surface + secondary-path）** — 完全空白
3. **End-to-end material-aware pointmap（模型自己预测哪些像素是非朗伯）** — 空白
4. **面向 pointmap 模型的非朗伯诊断 benchmark** — 空白

### 5.3 已占领方向（不要重复）

| 方向 | 占领者 | 时间 |
|------|--------|------|
| Pointmap uncertainty/置信度 | Trust3R | 2026.05 |
| Pointmap test-time adaptation | Free Geometry | 2026.04 |
| 镜像=stereo pair (单视图) | Reflect3r | 2025.09 |
| 3DGS 镜面双分支 | MirrorGaussian / Mirror-3DGS / SSR-GS / TR-Gaussians | 2024-2025 |
| 透明物体多层深度 (单目) | SeeGroup | 2026.05 |
| 偏振辅助单目深度 | CDPR | 2026.04 |
| Diffusion-based 透明深度 | DKT | 2025.12 |
| VGGT specular token suppression | EndoVGGT (医学限定) | 2026.03 |
| 传感器后处理修复非朗伯 | AnchorD | 2026.05 |
| 深度歧义 mixture density | Flying Points [2606.02552] | 2026.06 |

### 5.4 与本项目的切割定位

| 维度 | 本项目 (Layered Pointmap) | 最近竞品 |
|------|--------------------------|---------|
| 输入 | 多视图 RGB（≥2 帧） | Reflect3r：单视图+镜中反射 |
| 推理 | 单次前传 (feed-forward) | 3DGS 系列：per-scene 优化 |
| 输出 | First-surface + secondary-path pointmap + material mask | SeeGroup：多层深度（透明限定） |
| 覆盖材质 | 镜面、玻璃、金属、湿地面 | 各竞品仅覆盖一种 |
| 测试时先验 | 不需要 oracle mask/plane | MirrorGaussian 等需要 |
| 骨干 | VGGT-based | Reflect3r 用 DUSt3R |

### 5.5 主要 Baseline（实验必跑）

**Pointmap 路线**：
- VGGT (CVPR'25) — 骨干模型 baseline
- MASt3R / DUSt3R — 经典 baseline
- Trust3R — uncertainty 路线对照

**深度估计路线**：
- DepthAnything v2 / Marigold / DepthPro — mono depth
- FoundationStereo — stereo depth

**非朗伯专用**：
- Reflect3r — 同 setting 下必须直接对比
- SeeGroup — 多层深度对照
- AnchorD — sensor fusion 后处理对照

### 5.6 Benchmark 来源

- 自建合成数据 (~10k scenes, Blender 渲染带 non-Lambertian GT)
- 自建真实数据 (~50 scenes, reviewer 防御)
- Booster (CVPR'23) — stereo non-Lambertian
- Mirror3D / ScanNet++ mirror subset — 镜面
- ClearPose — 透明物体

---

## 六、新发现 & 原有调研更新

相比 2026-05 调研的新增/变化：

1. **VG²GT** [2606.01573]：VGGT + 3DGS 的结合方向出现了，但与非朗伯无关
2. **SeeGroup** [2605.28735]：多层深度的思路高度相关（permutation-invariant multi-layer），需要在 related work 中重点讨论并切割
3. **Flying Points (Depth Ambiguity)** [2606.02552]：mixture density 表征深度歧义，思路有交集但解决的是边界 flying point 问题而非非朗伯
4. **Anchor3R** [2606.05035]：streaming 3D 出现了，但是与非朗伯无关
5. **ViGeo** [2605.30060]：视频级时序一致的几何基础模型，潜在的更好骨干

---

## 七、结论

**核心发现**：截至 2026 年 6 月，"在 feed-forward pointmap 模型内部处理非朗伯几何失效"这个交叉点仍然是完全空白。最接近的工作是：
- Reflect3r（但限于单视图+平面镜设定）
- SeeGroup（但限于透明物体+单目深度，非 pointmap）
- Trust3R（标记不可信区域但不修复）

本项目的 Layered Pointmap 方案占据了一个独特且未被覆盖的位置。

---

## 双向链接

- [[../00-Project/Project-Overview]]
- [[Novelty-Check-2026-05-31]]
