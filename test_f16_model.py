#!/usr/bin/env python3
"""Quick test to verify F16Waypoint model size and functionality"""

import torch
import numpy as np
import gymnasium

# Add parent to path
import sys
sys.path.insert(0, '/workspace/PufferLib')

from pufferlib.ocean.torch import F16Waypoint, Recurrent

# Mock env matching F16Waypoint specs
class MockEnv:
    single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(28,), dtype=np.float32)
    single_action_space = gymnasium.spaces.Box(
        low=np.array([-2.0, -3.0, -1.0, 0.0], dtype=np.float32),
        high=np.array([6.0, 3.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32
    )

env = MockEnv()

print("=" * 60)
print("F16Waypoint Model Test")
print("=" * 60)

# Test standalone policy
print("\n1. Testing standalone F16Waypoint policy...")
model = F16Waypoint(env, hidden_size=1024)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
size_mb = (total_params * 4) / (1024 * 1024)

print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Model size (float32): {size_mb:.2f} MB")

# Test forward pass
batch_size = 32
obs = torch.randn(batch_size, 28)
logits, values = model(obs)

print(f"   Input shape: {obs.shape}")
print(f"   Output logits type: {type(logits)}")
print(f"   Output values shape: {values.shape}")
print(f"   Logits mean: {logits.mean}")
print(f"   Logits stddev: {logits.stddev}")

# Sample actions
actions = logits.sample()
print(f"   Sampled actions shape: {actions.shape}")
print(f"   Actions min/max: [{actions.min().item():.2f}, {actions.max().item():.2f}]")

# Test with LSTM wrapper
print("\n2. Testing with LSTM wrapper (Recurrent)...")
lstm_model = Recurrent(env, model, input_size=1024, hidden_size=1024)

total_params_lstm = sum(p.numel() for p in lstm_model.parameters())
size_mb_lstm = (total_params_lstm * 4) / (1024 * 1024)

print(f"   Total parameters (with LSTM): {total_params_lstm:,}")
print(f"   Model size (float32): {size_mb_lstm:.2f} MB")

# Test LSTM forward
state = {
    'lstm_h': None,
    'lstm_c': None,
}
obs_seq = torch.randn(batch_size, 28)
logits_lstm, values_lstm = lstm_model.forward_eval(obs_seq, state)

print(f"   LSTM output logits type: {type(logits_lstm)}")
print(f"   LSTM output values shape: {values_lstm.shape}")
print(f"   LSTM state h shape: {state['lstm_h'].shape}")
print(f"   LSTM state c shape: {state['lstm_c'].shape}")

print("\n3. Layer breakdown:")
print("-" * 70)
print(f"{'Layer':<45s} {'Shape':<20s} {'Params':>12s} {'Size (MB)':>10s}")
print("-" * 70)
for name, param in lstm_model.named_parameters():
    layer_size_mb = (param.numel() * 4) / (1024 * 1024)
    print(f"{name:<45s} {str(tuple(param.shape)):<20s} {param.numel():>12,} {layer_size_mb:>10.2f}")

print("=" * 60)
print("✓ All tests passed!")
print("=" * 60)
