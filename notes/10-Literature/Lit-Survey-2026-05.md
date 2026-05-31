---
tags: [literature, survey, non-lambertian, pointmap, mirror-reconstruction]
created: 2026-05-31
date_window: "2024-01 → 2026-05"
queries_run: 4
---

# 文献调研 — 非朗伯 Pointmap 重建 (2024-2026)

> 目的：在选定 contribution kernel 之前确认 *pointmap × non-Lambertian* 交集的真空白与撞车风险。

## 真空白（可做）

- **Feed-forward pointmap 路线 × mirror-aware 几何分解** —— 完全空白
- **Pointmap + material co-prediction (端到端 multi-task)** —— 空白
- **面向 pointmap-style 重建模型的 non-Lambertian benchmark** —— 完全空白

## 已被占领（不要重复做）

| 坑位 | 占领工作 | 时间 | 备注 |
|------|---------|------|------|
| Pointmap evidential uncertainty | [[Lit-Trust3R]] (Trust3R) | 2026.05 | NIW + 多视 t 分布 |
| Pointmap test-time adaptation | [[Lit-Free-Geometry]] (Free Geometry) | 2026.04 | Cross-view consistency LoRA |
| Pointmap 自标定 | LoRA3D | 2024.12 | 5 min 单卡 LoRA |
| Specular-aware token suppression | HD-VGGT | 2026.03 | 早期 transformer 层抑制 unstable token |
| 镜子反射作为辅助视角（**关键风险**） | [[Lit-Reflect3r-Risk]] | 2025.09 | 单视图+镜面=免费 stereo pair |
| NeRF/3DGS 镜面双分支分解 | [[Lit-NeRF-Mirror-Series]] | 2024-2026 | RefGaussian / Ref-Unlock / SSR-GS / Ref-DGS / RefGaussian |
| 多视图 intrinsic decomposition | IDT | 2025.12 | Diffuse + specular shading 分解 |
| 偏振辅助单目深度 | CDPR / Poppy | 2026.04 / 2026.03 | 物理模态加持 |
| 透明物体 video diffusion 深度 | DKT (TransPhy3D) | 2025.12 | "Diffusion knows transparency" |
| Diffusion + stereo metric 解 | GeoDiff | 2025.10 | 反问题框架 |
| 透明深度多层估计 | SeeGroup | 2026.05 | 排列不变多层 |
| Sensor-priorgrounding | AnchorD | 2026.05 | Factor-graph 训练-free 锚定 |

## 撞车风险矩阵 (针对原计划 [A] Real/Virtual 双 pointmap)

```
              [A1]    [A2]    [A3]    [B]
Reflect3r    中      低      低      低
NeRF mirror  低      低      极低    低  
SSR-GS       低      低      极低    低
IDT          极低    中      低      中
CDPR(polar)  极低    低      低      低
AnchorD      极低    低      中      极低
```

最终选择：**[A3] Benchmark + Method**，理由 dataset 部分对所有竞品免疫，method 部分受 Reflect3r 中度威胁但可通过 setting 切割。

## 关键 baseline 名单（method 阶段必跑）

- VGGT (CVPR'25) — 当代 pointmap 模型最强基线
- MASt3R / DUSt3R — 经典基线
- DepthAnything v2 / Marigold / DepthPro — mono depth 对照（即便范式不同也要给数字证明 pointmap 路线优势）
- IGEV-Stereo / FoundationStereo — stereo 路线对照（Booster 协议下 reviewer 会要）
- Reflect3r — 同 setting 下必须直接对比

## 关键 benchmark 名单（评估阶段必上）

- 自建合成 (~10k scenes) ← 主战场
- 自建小型真实 (~50 scenes) ← reviewer 防御
- Booster (CVPR'23) ← 传统 stereo non-Lambertian
- Mirror3D / Mirror-NeRF ← 镜面专用
- ScanNet++ mirror subset ← 大规模真实
- ClearPose ← 透明物体

## 引用关键

- 主威胁 (motivation 必须切割): Reflect3r [arxiv:2509.20607]
- NeRF 路线主对照: SSR-GS [arxiv:2603.05152], Ref-Unlock [arxiv:2507.06103]
- pointmap uncertainty 占位: Trust3R [arxiv:2605.19539]
- pointmap TTA 占位: Free Geometry [arxiv:2604.14048]
- 数据采集流程参考: AnchorD [arxiv:2605.02667] (喷哑光涂层 + 多相机融合)
- 透明合成数据参考: TransPhy3D / DKT [arxiv:2512.23705] (~11k 视频)

## 双向链接

- [[../00-Project/Project-Overview]]
- [[Round-Log-2026-05-30]]
