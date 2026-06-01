---
tags: [compute, environment, c500, kaggle, fallback-plan]
created: 2026-06-01
status: pending-c500-validation
---

# 算力策略 — Pilot 阶段

## 现状

| 平台 | 状态 | Pilot 适用度 |
|---|---|---|
| 本地 Mac MPS | ✅ 可用 | ❌ 仅能跑 inference debug |
| 曦云 C500 (沐曦) | ⚠️ 需验证 | ❓ 待 sanity check |
| H100 | ⏳ 未到货 | — |
| Kaggle 2×T4 | ✅ 可用 | ⚠️ Plan B (拼接 12h sessions) |
| Colab Pro A100 | ✅ 可用（如有 Pro） | ⚠️ Plan C (限时) |

## 第一性原理判断

**Pilot 的目的是 fail-fast，不是出最终结果**。所以可以用更小设置：
- 不需要 1B 全量 fine-tune，**冻结 encoder + 只训 heads** → 训练参数从 1B 降到 ~50-100M
- 不需要 batch=8，**batch=1-2 + gradient accumulation** 即可
- 不需要 H100，**T4 16GB / C500 都够 fp16 batch=1**
- Pilot 1k-2k 场景训练 ≤24h，对显卡要求大幅下降

## C500 兼容性 Sanity Check

### 怎么用
```bash
cd src/third_party/vggt
bash check_c500_compat.sh > c500_check.log 2>&1
cat c500_check.log
```

### 5 个关键验证点
| Step | 检查 | Pass 标准 |
|---|---|---|
| 1 | `nvidia-smi` 或 `mxsmi` 能列出 GPU | 看到 C500 |
| 2 | `torch.cuda.is_available()` | True (即使是 MACA/MUSA) |
| 2.5 | bf16 / fp16 matmul | 至少一个 OK |
| 3 | flash-attn (可选) | 不可用 OK，慢一点 |
| 4 | VGGT.from_pretrained() + forward | OK，记录显存峰值 |
| 5 | Backward pass (heads-only fine-tune) | OK，记录显存峰值 |

### 三种结果对应的下一步
- **全 OK** → 直接进 pilot Day 2-10
- **Step 4 OK, Step 5 OOM** → 启用 gradient checkpointing + 减 views (从 25 → 8)
- **Step 2 CUDA not available** → C500 没装 MACA-PyTorch，需联系运维或回退 Plan B

## Plan B — Kaggle (如果 C500 不行)

### 资源
- 2×T4 (16GB each) 或 1×P100 (16GB)
- 30h GPU/week, single session ≤ 12h
- 持久化数据集功能

### 调整
| 维度 | 原计划 (H100) | Kaggle 调整 |
|---|---|---|
| Batch | 4-8 | 1 + grad accum 8 |
| Views per scene | 25 | 8-12 |
| Mixed precision | bf16 | fp16 |
| Encoder | 部分 unfreeze | **全 freeze**, 只训 heads |
| Total scenes | 1k-2k | 1k (more iterations on less data) |
| Wall-clock | 24-36h | 60-80h (5-7 sessions × 12h) |
| 训练数据存储 | 本地 | Kaggle dataset (上传 1k 场景 ~50GB) |

### Pilot Kaggle 启动模板
```python
# Kaggle notebook 模板 (待填)
# 1. !git clone https://github.com/facebookresearch/vggt.git && cd vggt && pip install -e .
# 2. Mount Kaggle dataset (1k Blender scenes)
# 3. Resume training from previous session checkpoint (Kaggle output dataset)
# 4. Train for ~10h (留 2h margin), save checkpoint, reload next session
```

## Plan C — Colab Pro A100 (备选)

仅当 C500 + Kaggle 都不行时启用。$10/月 Pro 订阅 + A100 限时。

## 决策树

```
开始 pilot Day 1
│
├── 跑 check_c500_compat.sh
│   ├── 全 PASS → 用 C500 (主线)
│   ├── 仅 inference PASS → 修补 (gradient checkpoint / 减 views) → 重试
│   └── CUDA not available → Plan B
│
├── Plan B: Kaggle
│   ├── 上传数据集
│   ├── 适配 single-12h-session 训练循环
│   └── 5-7 个 session 后拿到 checkpoint
│
└── 任何情况下: 本地 Mac MPS 仅用于
    ├── 数据 pipeline 开发
    ├── Blender 合成脚本
    ├── 评估脚本
    └── 推理可视化 (fp32)
```

## 双向链接

- [[Pilot-Technical-Plan]]
- [[Experiment-Plan]]
- [[../00-Project/Project-Overview]]
