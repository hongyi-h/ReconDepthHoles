---
tags: [pilot, technical-plan, vggt, implementation]
created: 2026-05-31
status: ready-to-execute
gate: ≥15% Chamfer-NL reduction
---

# Pilot 技术方案 — VGGT Layered Head

## VGGT 代码结构（已验证）

```
facebookresearch/vggt (CVPR 2025 Best Paper, 13k+ stars)
├── vggt/
│   ├── heads/
│   │   ├── dpt_head.py      ← DPTHead: output_dim=4 (xyz+conf), multi-scale DPT decoder
│   │   ├── camera_head.py   ← pose prediction head
│   │   ├── track_head.py    ← tracking head
│   │   └── head_act.py      ← activation functions (inv_log, expp1)
│   ├── models/              ← main model (encoder + aggregation)
│   ├── layers/              ← transformer layers
│   └── utils/
├── training/
│   ├── trainer.py           ← 完整训练循环
│   ├── loss.py              ← MultitaskLoss (camera + depth + point)
│   ├── launch.py            ← distributed launch
│   ├── config/              ← yaml configs
│   └── data/                ← data loading
└── examples/                ← inference demos
```

## 改造方案（最小侵入）

### Step 1: 加 Secondary-Path Head

```python
# vggt/heads/secondary_dpt_head.py
# 直接复制 dpt_head.py，改名为 SecondaryDPTHead
# output_dim=4 (secondary xyz + confidence)
# 共享 encoder tokens，独立 DPT decoder weights
```

改动点：
- `vggt/models/vggt.py` 中加载第二个 DPTHead 实例
- forward 中对同一组 aggregated tokens 分别过 primary head 和 secondary head

### Step 2: 加 Mask Head

```python
# vggt/heads/mask_head.py
# 轻量 2-layer conv head
# 输入: aggregated tokens (same as DPT)
# 输出: per-pixel binary mask (non-Lambertian probability)
# Loss: BCE with GT material mask
```

### Step 3: 加 Symmetric Loss

```python
# training/symmetric_loss.py
def compute_symmetric_loss(primary_points, secondary_points, mirror_plane_gt):
    """
    对于 mask=1 的像素:
    secondary_point 应该是 primary_point 关于 mirror_plane 的镜像
    L_sym = ||secondary - Reflect(primary, plane)||_1
    """
    # mirror_plane: (B, 4) — normal (nx,ny,nz) + offset d
    # Reflect(p, plane) = p - 2 * (n·p + d) * n
    normal = mirror_plane_gt[:, :3]  # (B, 3)
    d = mirror_plane_gt[:, 3:]       # (B, 1)
    
    # (B, H, W, 3) dot (B, 1, 1, 3) → (B, H, W, 1)
    dot = (primary_points * normal[:, None, None, :]).sum(-1, keepdim=True)
    reflected = primary_points - 2 * (dot + d[:, None, None, :]) * normal[:, None, None, :]
    
    loss = F.l1_loss(secondary_points, reflected, reduction='none')
    return loss  # masked externally
```

### Step 4: 修改 MultitaskLoss

```python
# training/loss.py — 在 MultitaskLoss.forward 中加:
if "secondary_world_points" in predictions:
    sec_loss_dict = compute_point_loss(predictions_secondary, batch, **self.secondary_point)
    sym_loss_dict = compute_symmetric_loss(...)
    total_loss += sec_loss * weight_sec + sym_loss * weight_sym
```

## 数据准备方案

### 合成数据 Pipeline (Blender)

```
每个场景生成:
├── RGB images (10-30 views, 640×480)
├── first_surface_pointmap.npy (H, W, 3) — 光线第一次命中的 3D 点
├── secondary_pointmap.npy (H, W, 3) — 光线经镜面反射后命中的 3D 点 (镜面区域)
├── material_mask.npy (H, W) — 0=Lambertian, 1=mirror, 2=glass, 3=glossy
├── mirror_plane.npy (4,) — 镜面平面参数 [nx, ny, nz, d]
├── camera_intrinsics.npy
└── camera_extrinsics.npy

Blender 脚本核心:
- 用 Cycles ray-tracing
- 第一次 hit = first_surface (bpy.context.scene.cycles.max_bounces = 0 for depth pass)
- 镜面反射 hit = secondary (trace reflected ray manually via Python API)
- Material mask: 从 shader node 类型判断
```

### 3DReflecNet Reflective Subset 借用

```bash
# 下载 reflective 类目 (~5k objects)
# 转换为 multi-view pointmap 格式:
# - 已有 depth + camera → 可直接 unproject 为 pointmap
# - 但没有 secondary-path GT → 只能用于 primary head 训练 / 评估
# - 用途: 增强 primary head 在 specular 物体上的鲁棒性
```

## Pilot 时间线 (10 天)

| Day | 任务 | 产出 |
|---|---|---|
| 1 | Clone VGGT, 环境搭建, 跑通 inference | 确认 checkpoint 可加载 |
| 2 | 写 Blender mirror-scene 生成脚本 | 生成 100 场景验证格式 |
| 3 | 扩展到 1k 场景 + 数据 loader | train/val/test split ready |
| 4 | 实现 SecondaryDPTHead + MaskHead | 模型可 forward |
| 5 | 实现 symmetric_loss + 修改 MultitaskLoss | 训练可启动 |
| 6-7 | 训练 (2×H100, ~24-36h) | checkpoint |
| 8 | 评估: Chamfer-NL / Chamfer-L / mask accuracy | 数字 |
| 9 | Ab1 ablation: single-head baseline (same data) | 对比数字 |
| 10 | 决策 + 写 Pilot-Results.md | PROCEED / ITERATE / PIVOT |

## 评估指标定义

```python
# Chamfer-NL: Chamfer distance on non-Lambertian masked pixels only
# Chamfer-L: Chamfer distance on Lambertian masked pixels only
# Chamfer-Virtual: Chamfer distance of secondary_pred vs secondary_gt (mirror region only)

def chamfer_masked(pred_points, gt_points, mask):
    """
    pred_points: (B, H, W, 3)
    gt_points: (B, H, W, 3)
    mask: (B, H, W) binary
    """
    valid = mask.bool()
    pred_flat = pred_points[valid]  # (N, 3)
    gt_flat = gt_points[valid]      # (N, 3)
    return F.l1_loss(pred_flat, gt_flat)  # 简化版; 正式版用 chamfer_distance
```

## GATE 判定标准

```
PROCEED:  Chamfer-NL 降低 ≥15% vs VGGT-zero-shot AND Chamfer-L 退化 <2%
ITERATE:  10-15% 降低 → 调 loss weights / 加数据 / 换 head 结构
PIVOT:    <10% 降低 → 回 deep-interview 重选 kernel
```

## 双向链接

- [[Experiment-Plan]]
- [[../../00-Project/Project-Overview]]
- [[Pilot-Results]] — (待填)
