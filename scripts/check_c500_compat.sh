#!/bin/bash
# ============================================================
# C500 Compatibility Sanity Check for VGGT Pilot
# 在曦云 C500 上跑这个脚本，10 分钟内确认能否训练 VGGT
#
# 用法: 在 C500 上 clone 项目后:
#   cd ReconDepthHoles
#   pip install -e src/third_party/vggt  # 先装 vggt
#   bash scripts/check_c500_compat.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VGGT_DIR="$PROJECT_ROOT/src/third_party/vggt"

# 确保 vggt 在 Python path 中
export PYTHONPATH="$VGGT_DIR:$PYTHONPATH"
export VGGT_DIR
cd "$VGGT_DIR"

echo "=== Step 1: GPU 基本信息 ==="
# 如果是 MACA/MUSA 环境，可能没有 nvidia-smi
# 尝试多种方式获取 GPU 信息
nvidia-smi 2>/dev/null || mxsmi 2>/dev/null || echo "Neither nvidia-smi nor mxsmi found"

echo ""
echo "=== Step 2: PyTorch + CUDA/MACA 兼容性 ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        mem_bytes = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f'  Memory: {mem_bytes / 1e9:.1f} GB')
        cap = torch.cuda.get_device_capability(i)
        print(f'  Compute capability: {cap[0]}.{cap[1]}')
    # bf16 支持测试
    try:
        x = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
        y = x @ x.T
        print(f'  BF16 matmul: OK')
    except Exception as e:
        print(f'  BF16 matmul: FAILED ({e})')
    # fp16 支持测试
    try:
        x = torch.randn(2, 2, device='cuda', dtype=torch.float16)
        y = x @ x.T
        print(f'  FP16 matmul: OK')
    except Exception as e:
        print(f'  FP16 matmul: FAILED ({e})')
else:
    print('CUDA NOT available - VGGT training will NOT work')
    print('Check if MACA/MUSA driver is installed and torch is compiled with it')
"

echo ""
echo "=== Step 3: Flash Attention 兼容性 (可选但影响速度) ==="
python3 -c "
try:
    from flash_attn import flash_attn_func
    print('flash-attn: available')
except ImportError:
    print('flash-attn: NOT available (will use standard attention, slower but OK)')
"

echo ""
echo "=== Step 4: VGGT 模型加载测试 ==="
python3 -c "
import torch
import sys
import os; sys.path.insert(0, os.environ.get('VGGT_DIR', '.'))

try:
    from vggt.models.vggt import VGGT
    print('VGGT import: OK')
except ImportError as e:
    print(f'VGGT import: FAILED ({e})')
    print('Run: pip install -e . first')
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

if device == 'cpu':
    print('ABORT: Cannot test GPU loading on CPU')
    sys.exit(1)

# 尝试加载模型到 GPU
try:
    model = VGGT.from_pretrained('facebook/VGGT-1B')
    model = model.to(device)
    print(f'Model loaded to {device}: OK')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')

    # 测试 forward pass (小 batch)
    dummy = torch.randn(2, 3, 350, 518, device=device)  # 2 views
    with torch.no_grad():
        # 用 fp32 先测
        out = model(dummy.unsqueeze(0))  # add batch dim
    print(f'Forward pass: OK')
    print(f'Output keys: {list(out.keys())}')

    # 显存占用
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'Peak GPU memory (2 views, fp32): {mem:.1f} GB')

except Exception as e:
    print(f'Model loading/forward FAILED: {e}')
    import traceback
    traceback.print_exc()

# 清理
torch.cuda.empty_cache()
"

echo ""
echo "=== Step 5: 训练可行性快速测试 (backward pass) ==="
python3 -c "
import torch
import sys
import os; sys.path.insert(0, os.environ.get('VGGT_DIR', '.'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('SKIP: no GPU')
    sys.exit(0)

from vggt.models.vggt import VGGT

model = VGGT.from_pretrained('facebook/VGGT-1B')
model = model.to(device)
model.train()

# 只解冻最后几层 + head (模拟 pilot fine-tune)
for param in model.parameters():
    param.requires_grad = False

# 解冻 point head
for name, param in model.named_parameters():
    if 'point' in name or 'dpt' in name:
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable/1e6:.1f}M / {total/1e9:.2f}B total')

# 测试 backward
dummy = torch.randn(2, 3, 350, 518, device=device)
try:
    out = model(dummy.unsqueeze(0))
    # 假设 world_points 是 pointmap 输出
    if 'world_points_list' in out:
        loss = out['world_points_list'][-1].sum()
    elif 'world_points' in out:
        loss = out['world_points'].sum()
    else:
        loss = list(out.values())[0].sum()
    loss.backward()
    print(f'Backward pass: OK')
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'Peak GPU memory (train, 2 views): {mem:.1f} GB')
except Exception as e:
    print(f'Backward FAILED: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== SUMMARY ==="
echo "如果 Step 4 和 Step 5 都 OK → C500 可以跑 VGGT pilot"
echo "如果 Step 4 OK 但 Step 5 OOM → 需要减少 views 或用 gradient checkpointing"
echo "如果 Step 2 CUDA not available → 需要安装 MACA/MUSA 版 PyTorch"
echo "把这个输出发给我，我帮你判断下一步"
