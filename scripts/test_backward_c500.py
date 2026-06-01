"""
Pilot sanity check: verify backward pass works on C500 by disabling gradient checkpointing.
64GB per card is enough to run without it.

Usage (on C500):
    cd ReconDepthHoles
    python scripts/test_backward_c500.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'third_party', 'vggt'))

from vggt.models.vggt import VGGT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', "This script requires GPU"

print("Loading VGGT...")
model = VGGT.from_pretrained('facebook/VGGT-1B')
model = model.to(device)

# Disable gradient checkpointing by keeping model in eval mode for the aggregator
# but enabling gradients on the heads we want to train.
# This avoids the MetaX torch.utils.checkpoint bug while still allowing backward.
model.eval()  # keeps aggregator in eval path (no checkpoint calls)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze point head (DPT head for pointmap)
unfrozen_count = 0
for name, param in model.named_parameters():
    if 'point' in name or 'dpt' in name:
        param.requires_grad = True
        unfrozen_count += 1

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable/1e6:.1f}M / {total/1e9:.2f}B total ({unfrozen_count} tensors)')

# Forward + backward test
print("Testing forward + backward (2 views, 350x518)...")
dummy = torch.randn(1, 2, 3, 350, 518, device=device)  # (B=1, S=2, C=3, H, W)

# Enable grad computation even though model is in eval mode
with torch.enable_grad():
    out = model(dummy)

print(f'Output keys: {list(out.keys())}')

# Find pointmap output and compute dummy loss
if 'world_points_list' in out:
    target = out['world_points_list'][-1]
elif 'world_points' in out:
    target = out['world_points']
else:
    target = list(out.values())[0]

loss = target.sum()
loss.backward()
print(f'Backward pass: OK')

mem = torch.cuda.max_memory_allocated() / 1e9
print(f'Peak GPU memory: {mem:.1f} GB')

# Verify gradients exist on unfrozen params
grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
print(f'Parameters with gradients: {grad_count}/{unfrozen_count}')

if grad_count > 0:
    print("\n✅ C500 BACKWARD PASS VERIFIED. Ready for pilot training.")
else:
    print("\n❌ No gradients computed. Something is wrong.")

torch.cuda.empty_cache()
