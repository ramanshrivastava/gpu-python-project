"""
GPU utilities for device management and performance benchmarking.
"""

import time
import torch
import numpy as np
from typing import Tuple, Optional


def get_device() -> torch.device:
    """
    Get the best available device (GPU if available, otherwise CPU).
    Supports CUDA (NVIDIA) and MPS (Apple Silicon) backends.
    
    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ NVIDIA GPU available: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA version: {torch.version.cuda}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Apple Silicon GPU available: Metal Performance Shaders (MPS)")
        print(f"🍎 Running on Apple Silicon with GPU acceleration")
        print(f"🔧 PyTorch MPS backend: {torch.backends.mps.is_built()}")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU acceleration available, using CPU")
    
    return device


def benchmark_matrix_multiplication(
    matrix_size: int = 2048, 
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Benchmark matrix multiplication on CPU vs GPU.
    
    Args:
        matrix_size: Size of the square matrices to multiply
        device: Device to use for GPU benchmark
    
    Returns:
        Tuple of (cpu_time, gpu_time) in seconds
    """
    if device is None:
        device = get_device()
    
    # Create random matrices
    np.random.seed(42)
    a_np = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    b_np = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    
    # CPU benchmark
    print(f"🔄 Benchmarking {matrix_size}x{matrix_size} matrix multiplication...")
    
    start_time = time.time()
    _ = np.dot(a_np, b_np)
    cpu_time = time.time() - start_time
    print(f"🖥️  CPU time: {cpu_time:.4f} seconds")
    
    # GPU benchmark (if available)
    gpu_time = float('inf')
    if device.type in ['cuda', 'mps']:
        a_gpu = torch.from_numpy(a_np).to(device)
        b_gpu = torch.from_numpy(b_np).to(device)
        
        # Warm up
        _ = torch.mm(a_gpu, b_gpu)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        _ = torch.mm(a_gpu, b_gpu)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        gpu_time = time.time() - start_time
        
        gpu_name = "CUDA GPU" if device.type == 'cuda' else "Apple Silicon GPU"
        print(f"🚀 {gpu_name} time: {gpu_time:.4f} seconds")
        print(f"⚡ Speedup: {cpu_time / gpu_time:.2f}x")
    
    return cpu_time, gpu_time


def memory_info() -> dict:
    """
    Get GPU memory information.
    Supports both CUDA and MPS backends.
    
    Returns:
        Dictionary with memory statistics
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        return {
            "backend": "CUDA",
            "allocated_gb": allocated,
            "cached_gb": cached,
            "max_allocated_gb": max_allocated,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    elif torch.backends.mps.is_available():
        # MPS doesn't have the same detailed memory tracking as CUDA
        return {
            "backend": "MPS (Apple Silicon)",
            "status": "Memory tracking limited on MPS backend",
            "note": "Use Activity Monitor to check GPU usage"
        }
    else:
        return {"error": "No GPU backend available"}


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 CUDA GPU memory cache cleared")
    elif torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing, but we can try to free memory
        print("🧹 MPS backend: Memory management handled automatically")
        print("   Tip: Use torch.mps.empty_cache() if available in future PyTorch versions")
    else:
        print("⚠️  No GPU available to clear memory")


if __name__ == "__main__":
    # Quick test
    device = get_device()
    benchmark_matrix_multiplication(1024, device)
    
    if device.type in ['cuda', 'mps']:
        print("\n📊 GPU Memory Info:")
        mem_info = memory_info()
        for key, value in mem_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
