#!/usr/bin/env python3
"""
Intense GPU Stress Test for Apple Silicon
Run this while watching macmon to see GPU spike.

Usage:
    Terminal 1: macmon
    Terminal 2: python gpu_stress_test.py
"""

import torch
import time
import sys


def stress_test_matmul(device: torch.device, size: int = 8192, iterations: int = 100):
    """Sustained matrix multiplication stress test."""
    print(f"\n🔥 Matrix Multiplication Stress Test")
    print(f"   Size: {size}x{size}, Iterations: {iterations}")
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    _ = torch.mm(a, b)
    torch.mps.synchronize()
    
    start = time.time()
    for i in range(iterations):
        c = torch.mm(a, b)
        if (i + 1) % 10 == 0:
            torch.mps.synchronize()
            elapsed = time.time() - start
            ops_per_sec = (i + 1) / elapsed
            print(f"   Progress: {i+1}/{iterations} ({ops_per_sec:.1f} ops/sec)", end="\r")
    
    torch.mps.synchronize()
    total_time = time.time() - start
    print(f"\n   ✅ Completed in {total_time:.2f}s ({iterations/total_time:.1f} ops/sec)")


def stress_test_conv(device: torch.device, iterations: int = 500):
    """Convolution stress test (simulates CNN forward pass)."""
    print(f"\n🔥 Convolution Stress Test")
    print(f"   Batch: 64, Channels: 256→512, Iterations: {iterations}")
    
    # Simulate a heavy conv layer
    conv = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1).to(device)
    x = torch.randn(64, 256, 56, 56, device=device)
    
    # Warmup
    _ = conv(x)
    torch.mps.synchronize()
    
    start = time.time()
    for i in range(iterations):
        y = conv(x)
        if (i + 1) % 50 == 0:
            torch.mps.synchronize()
            elapsed = time.time() - start
            print(f"   Progress: {i+1}/{iterations} ({(i+1)/elapsed:.1f} ops/sec)", end="\r")
    
    torch.mps.synchronize()
    total_time = time.time() - start
    print(f"\n   ✅ Completed in {total_time:.2f}s")


def stress_test_transformer(device: torch.device, iterations: int = 200):
    """Transformer attention stress test."""
    print(f"\n🔥 Transformer Attention Stress Test")
    print(f"   Batch: 32, Seq: 512, Heads: 16, Dim: 1024")
    
    batch, seq_len, num_heads, head_dim = 32, 512, 16, 64
    embed_dim = num_heads * head_dim
    
    # Multi-head attention
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
    x = torch.randn(batch, seq_len, embed_dim, device=device)
    
    # Warmup
    _ = mha(x, x, x)
    torch.mps.synchronize()
    
    start = time.time()
    for i in range(iterations):
        out, _ = mha(x, x, x)
        if (i + 1) % 20 == 0:
            torch.mps.synchronize()
            elapsed = time.time() - start
            print(f"   Progress: {i+1}/{iterations} ({(i+1)/elapsed:.1f} ops/sec)", end="\r")
    
    torch.mps.synchronize()
    total_time = time.time() - start
    print(f"\n   ✅ Completed in {total_time:.2f}s")


def stress_test_memory(device: torch.device, gb: float = 20.0):
    """Memory allocation stress test."""
    print(f"\n🔥 Memory Stress Test")
    print(f"   Allocating ~{gb}GB of tensors...")
    
    tensors = []
    allocated = 0
    target_bytes = int(gb * 1024 ** 3)
    chunk_size = 1024  # 1024 x 1024 x 256 floats = 1GB per tensor
    
    start = time.time()
    try:
        while allocated < target_bytes:
            t = torch.randn(1024, 1024, 256, device=device)
            tensors.append(t)
            allocated += t.numel() * 4  # float32 = 4 bytes
            print(f"   Allocated: {allocated / (1024**3):.1f} GB", end="\r")
    except RuntimeError as e:
        print(f"\n   ⚠️  Hit memory limit at {allocated / (1024**3):.1f} GB")
    
    # Do some operations on the tensors
    print(f"\n   Running operations on {len(tensors)} tensors...")
    for i, t in enumerate(tensors[:10]):  # Just first 10
        _ = t * 2 + 1
        torch.mps.synchronize()
    
    total_time = time.time() - start
    print(f"   ✅ Memory test completed in {total_time:.2f}s")
    
    # Cleanup
    del tensors
    torch.mps.empty_cache()


def sustained_load(device: torch.device, duration_seconds: int = 30):
    """Sustained GPU load for specified duration."""
    print(f"\n🔥 SUSTAINED LOAD TEST ({duration_seconds} seconds)")
    print(f"   This will max out your GPU - watch macmon!")
    print(f"   Press Ctrl+C to stop early\n")
    
    # Large matrices for maximum compute
    size = 8192
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start = time.time()
    ops = 0
    
    try:
        while (time.time() - start) < duration_seconds:
            c = torch.mm(a, b)
            ops += 1
            if ops % 5 == 0:
                torch.mps.synchronize()
                elapsed = time.time() - start
                remaining = duration_seconds - elapsed
                print(f"   ⏱️  {elapsed:.1f}s elapsed | {remaining:.1f}s remaining | {ops} ops", end="\r")
    except KeyboardInterrupt:
        print("\n   Stopped by user")
    
    torch.mps.synchronize()
    total_time = time.time() - start
    print(f"\n   ✅ Sustained load completed: {ops} operations in {total_time:.1f}s")


def main():
    print("=" * 60)
    print("  🍎 APPLE SILICON GPU STRESS TEST")
    print("=" * 60)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available!")
        sys.exit(1)
    
    device = torch.device("mps")
    print(f"\n✅ Using device: {device}")
    print(f"💡 TIP: Run 'macmon' in another terminal to watch GPU usage!\n")
    
    input("Press Enter to start stress tests...")
    
    # Run all stress tests
    stress_test_matmul(device, size=8192, iterations=50)
    stress_test_conv(device, iterations=300)
    stress_test_transformer(device, iterations=100)
    
    # Optional: sustained load
    print("\n" + "-" * 60)
    response = input("Run 30-second sustained load test? (y/n): ")
    if response.lower() == 'y':
        sustained_load(device, duration_seconds=30)
    
    # Optional: memory test
    response = input("Run memory stress test (~20GB)? (y/n): ")
    if response.lower() == 'y':
        stress_test_memory(device, gb=20.0)
    
    print("\n" + "=" * 60)
    print("  ✅ All stress tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
