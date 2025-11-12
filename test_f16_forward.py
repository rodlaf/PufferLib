#!/usr/bin/env python3
"""Test F16Waypoint model forward pass to debug -0.2 return issue"""

import sys
sys.path.insert(0, '/workspace/PufferLib')

import torch
import numpy as np
import gymnasium
import os
import tempfile

from pufferlib.ocean.torch import F16Waypoint

class MockEnv:
    single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(28,), dtype=np.float32)
    single_action_space = gymnasium.spaces.Box(
        low=np.array([-2.0, -3.0, -1.0, 0.0], dtype=np.float32),
        high=np.array([6.0, 3.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32
    )

print("="*60)
print("F16Waypoint Forward Pass Test")
print("="*60)

env = MockEnv()
model = F16Waypoint(env, hidden_size=128)

print(f"\nModel created successfully")
print(f"  hidden_size: 128")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_size_mb = (total_params * 4) / (1024 * 1024)

print(f"  Parameters: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Params size (float32): {params_size_mb:.2f} MB")

# Test actual saved model size
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
    tmp_path = tmp.name
    torch.save(model.state_dict(), tmp_path)
    file_size_bytes = os.path.getsize(tmp_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"  Saved model size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    os.unlink(tmp_path)  # Clean up

# Test forward pass
batch_size = 32
obs = torch.randn(batch_size, 28)

print(f"\nTesting forward pass with batch_size={batch_size}...")

try:
    logits, values = model(obs)
    
    print("✓ Forward pass successful")

    # Check for NaNs
    mean_has_nan = torch.isnan(logits.mean).any()
    std_has_nan = torch.isnan(logits.stddev).any()
    values_has_nan = torch.isnan(values).any()
    
    if mean_has_nan or std_has_nan or values_has_nan:
        print(f"\n✗ NaN DETECTED!")
        print(f"  Mean has NaN: {mean_has_nan}")
        print(f"  Std has NaN: {std_has_nan}")
        print(f"  Values has NaN: {values_has_nan}")
    else:
        print(f"\n✓ No NaNs in outputs")
    
    # Sample actions
    actions = logits.sample()
    
    # Check if actions are in valid range
    action_space_low = torch.tensor([-2.0, -3.0, -1.0, 0.0])
    action_space_high = torch.tensor([6.0, 3.0, 1.0, 1.0])
    
    print(f"\nAction space bounds: [{action_space_low.tolist()}, {action_space_high.tolist()}]")
    
    # Test multiple forward passes to check for instability
    print(f"\nTesting stability over 10 forward passes...")
    for i in range(10):
        obs = torch.randn(batch_size, 28)
        logits, values = model(obs)
        actions = logits.sample()
        
        if torch.isnan(actions).any():
            print(f"  ✗ Iteration {i}: NaN detected!")
            break
    else:
        print(f"  ✓ All 10 iterations stable")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
