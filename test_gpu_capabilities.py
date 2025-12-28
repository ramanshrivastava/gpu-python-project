#!/usr/bin/env python3
"""
Comprehensive Mac GPU/Metal Capabilities Test
Tests all available GPU acceleration options on macOS.
"""

import subprocess
import sys
import time
import json
from typing import Dict, Any, Optional


def run_cmd(command: str) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_item(key: str, value: Any, indent: int = 2):
    """Print formatted key-value pair."""
    spaces = " " * indent
    if isinstance(value, bool):
        icon = "✅" if value else "❌"
        print(f"{spaces}{icon} {key}: {value}")
    else:
        print(f"{spaces}• {key}: {value}")


# =============================================================================
# 1. HARDWARE DETECTION
# =============================================================================
def get_hardware_info() -> Dict[str, Any]:
    """Get Mac hardware specifications."""
    info = {}
    
    # Get chip info
    hw_output = run_cmd("system_profiler SPHardwareDataType")
    for line in hw_output.split('\n'):
        line = line.strip()
        if line.startswith("Chip:"):
            info["chip"] = line.split(":", 1)[1].strip()
        elif line.startswith("Total Number of Cores:"):
            info["cpu_cores"] = line.split(":", 1)[1].strip()
        elif line.startswith("Memory:"):
            info["memory"] = line.split(":", 1)[1].strip()
        elif line.startswith("Model Name:"):
            info["model"] = line.split(":", 1)[1].strip()
    
    # Get GPU info
    gpu_output = run_cmd("system_profiler SPDisplaysDataType")
    for line in gpu_output.split('\n'):
        line = line.strip()
        if "Chipset Model:" in line:
            info["gpu_chipset"] = line.split(":", 1)[1].strip()
        elif "Metal Support:" in line:
            info["metal_support"] = line.split(":", 1)[1].strip()
        elif "Total Number of Cores:" in line and "gpu_cores" not in info:
            info["gpu_cores"] = line.split(":", 1)[1].strip()
    
    return info


# =============================================================================
# 2. METAL API TEST
# =============================================================================
def test_metal_api() -> Dict[str, Any]:
    """Test Metal API availability using system tools."""
    results = {"available": False}
    
    # Check if Metal is available via system_profiler
    gpu_output = run_cmd("system_profiler SPDisplaysDataType")
    if "Metal Support:" in gpu_output:
        for line in gpu_output.split('\n'):
            if "Metal Support:" in line:
                metal_version = line.split(":", 1)[1].strip()
                results["available"] = True
                results["version"] = metal_version
                break
    
    # Check Metal device count using a simple Swift snippet
    metal_check = run_cmd("""
        swift -e 'import Metal; print(MTLCopyAllDevices().count)'
    """)
    if metal_check.isdigit():
        results["device_count"] = int(metal_check)
    
    return results


# =============================================================================
# 3. PYTORCH MPS TEST
# =============================================================================
def test_pytorch_mps() -> Dict[str, Any]:
    """Test PyTorch MPS (Metal Performance Shaders) backend."""
    results = {"installed": False}
    
    try:
        import torch
        results["installed"] = True
        results["version"] = torch.__version__
        results["mps_built"] = torch.backends.mps.is_built()
        results["mps_available"] = torch.backends.mps.is_available()
        
        if results["mps_available"]:
            # Test actual tensor operations
            try:
                device = torch.device("mps")
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                torch.mps.synchronize()
                results["mps_functional"] = True
                results["test_passed"] = True
            except Exception as e:
                results["mps_functional"] = False
                results["error"] = str(e)
    except ImportError:
        results["error"] = "PyTorch not installed"
    
    return results


# =============================================================================
# 4. MLX TEST (Apple's ML Framework)
# =============================================================================
def test_mlx() -> Dict[str, Any]:
    """Test Apple MLX framework availability."""
    results = {"installed": False}
    
    try:
        import mlx.core as mx
        results["installed"] = True
        results["version"] = mx.__version__ if hasattr(mx, '__version__') else "unknown"
        
        # Test basic operations
        try:
            x = mx.array([1.0, 2.0, 3.0])
            y = mx.array([4.0, 5.0, 6.0])
            z = mx.add(x, y)
            mx.eval(z)
            results["functional"] = True
            
            # Check default device
            results["default_device"] = str(mx.default_device())
            results["gpu_available"] = "gpu" in str(mx.default_device()).lower()
        except Exception as e:
            results["functional"] = False
            results["error"] = str(e)
    except ImportError:
        results["error"] = "MLX not installed (pip install mlx)"
    
    return results


# =============================================================================
# 5. TENSORFLOW METAL TEST
# =============================================================================
def test_tensorflow_metal() -> Dict[str, Any]:
    """Test TensorFlow with Metal plugin."""
    results = {"installed": False}
    
    try:
        import tensorflow as tf
        results["installed"] = True
        results["version"] = tf.__version__
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        results["gpu_count"] = len(gpus)
        results["gpu_available"] = len(gpus) > 0
        
        if gpus:
            results["gpu_devices"] = [gpu.name for gpu in gpus]
            
            # Test actual computation
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([100, 100])
                    b = tf.random.normal([100, 100])
                    c = tf.matmul(a, b)
                results["functional"] = True
            except Exception as e:
                results["functional"] = False
                results["error"] = str(e)
    except ImportError:
        results["error"] = "TensorFlow not installed"
    
    return results


# =============================================================================
# 6. JAX METAL TEST
# =============================================================================
def test_jax_metal() -> Dict[str, Any]:
    """Test JAX with Metal backend."""
    results = {"installed": False}
    
    try:
        import jax
        results["installed"] = True
        results["version"] = jax.__version__
        
        # Check devices
        devices = jax.devices()
        results["device_count"] = len(devices)
        results["devices"] = [str(d) for d in devices]
        results["default_backend"] = str(jax.default_backend())
        results["gpu_available"] = any("gpu" in str(d).lower() or "metal" in str(d).lower() for d in devices)
        
        # Test computation
        try:
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            z = jnp.dot(x, y)
            results["functional"] = True
        except Exception as e:
            results["functional"] = False
            results["error"] = str(e)
    except ImportError:
        results["error"] = "JAX not installed"
    
    return results


# =============================================================================
# 7. PERFORMANCE BENCHMARK
# =============================================================================
def run_benchmark() -> Dict[str, Any]:
    """Run performance benchmarks across available backends."""
    results = {}
    
    try:
        import torch
        if torch.backends.mps.is_available():
            print("\n  Running PyTorch MPS benchmark...")
            
            sizes = [512, 1024, 2048, 4096]
            for size in sizes:
                # CPU benchmark
                a_cpu = torch.randn(size, size)
                b_cpu = torch.randn(size, size)
                
                start = time.perf_counter()
                _ = torch.mm(a_cpu, b_cpu)
                cpu_time = time.perf_counter() - start
                
                # MPS benchmark
                device = torch.device("mps")
                a_mps = a_cpu.to(device)
                b_mps = b_cpu.to(device)
                
                # Warmup
                _ = torch.mm(a_mps, b_mps)
                torch.mps.synchronize()
                
                start = time.perf_counter()
                _ = torch.mm(a_mps, b_mps)
                torch.mps.synchronize()
                mps_time = time.perf_counter() - start
                
                speedup = cpu_time / mps_time if mps_time > 0 else 0
                results[f"matmul_{size}x{size}"] = {
                    "cpu_ms": round(cpu_time * 1000, 3),
                    "mps_ms": round(mps_time * 1000, 3),
                    "speedup": round(speedup, 2)
                }
    except Exception as e:
        results["pytorch_error"] = str(e)
    
    # MLX benchmark
    try:
        import mlx.core as mx
        print("  Running MLX benchmark...")
        
        for size in [512, 1024, 2048, 4096]:
            a = mx.random.normal((size, size))
            b = mx.random.normal((size, size))
            
            # Warmup
            c = mx.matmul(a, b)
            mx.eval(c)
            
            start = time.perf_counter()
            c = mx.matmul(a, b)
            mx.eval(c)
            mlx_time = time.perf_counter() - start
            
            key = f"matmul_{size}x{size}"
            if key in results:
                results[key]["mlx_ms"] = round(mlx_time * 1000, 3)
            else:
                results[key] = {"mlx_ms": round(mlx_time * 1000, 3)}
    except Exception as e:
        results["mlx_error"] = str(e)
    
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "🖥️  MAC GPU/METAL CAPABILITIES TEST ".center(60, "="))
    print(f"  Running comprehensive GPU acceleration tests...")
    
    all_results = {}
    
    # 1. Hardware
    print_section("1. HARDWARE SPECIFICATIONS")
    hw_info = get_hardware_info()
    all_results["hardware"] = hw_info
    for k, v in hw_info.items():
        print_item(k.replace("_", " ").title(), v)
    
    # 2. Metal API
    print_section("2. METAL API")
    metal_info = test_metal_api()
    all_results["metal"] = metal_info
    for k, v in metal_info.items():
        print_item(k.replace("_", " ").title(), v)
    
    # 3. PyTorch MPS
    print_section("3. PYTORCH MPS")
    pytorch_info = test_pytorch_mps()
    all_results["pytorch_mps"] = pytorch_info
    for k, v in pytorch_info.items():
        print_item(k.replace("_", " ").title(), v)
    
    # 4. MLX
    print_section("4. APPLE MLX")
    mlx_info = test_mlx()
    all_results["mlx"] = mlx_info
    for k, v in mlx_info.items():
        print_item(k.replace("_", " ").title(), v)
    
    # 5. TensorFlow
    print_section("5. TENSORFLOW METAL")
    tf_info = test_tensorflow_metal()
    all_results["tensorflow"] = tf_info
    for k, v in tf_info.items():
        print_item(k.replace("_", " ").title(), v)
    
    # 6. JAX
    print_section("6. JAX")
    jax_info = test_jax_metal()
    all_results["jax"] = jax_info
    for k, v in jax_info.items():
        print_item(k.replace("_", " ").title(), v)
    
    # 7. Benchmarks
    print_section("7. PERFORMANCE BENCHMARKS")
    bench_results = run_benchmark()
    all_results["benchmarks"] = bench_results
    
    print("\n  Matrix Multiplication (lower ms = faster):")
    print(f"  {'Size':<12} {'CPU (ms)':<12} {'MPS (ms)':<12} {'MLX (ms)':<12} {'Speedup':<10}")
    print(f"  {'-'*56}")
    
    for key in sorted(bench_results.keys()):
        if key.startswith("matmul_"):
            size = key.replace("matmul_", "")
            data = bench_results[key]
            cpu = data.get("cpu_ms", "N/A")
            mps = data.get("mps_ms", "N/A")
            mlx = data.get("mlx_ms", "N/A")
            speedup = data.get("speedup", "N/A")
            print(f"  {size:<12} {str(cpu):<12} {str(mps):<12} {str(mlx):<12} {str(speedup)+'x' if speedup != 'N/A' else 'N/A':<10}")
    
    # Summary
    print_section("SUMMARY")
    gpu_options = []
    if pytorch_info.get("mps_available"):
        gpu_options.append("PyTorch MPS")
    if mlx_info.get("functional"):
        gpu_options.append("Apple MLX")
    if tf_info.get("gpu_available"):
        gpu_options.append("TensorFlow Metal")
    if jax_info.get("gpu_available"):
        gpu_options.append("JAX Metal")
    
    if gpu_options:
        print(f"  ✅ GPU acceleration available via: {', '.join(gpu_options)}")
    else:
        print("  ❌ No GPU acceleration detected")
    
    print(f"  💾 Unified Memory: {hw_info.get('memory', 'Unknown')}")
    print(f"  🎮 Metal Version: {metal_info.get('version', 'Unknown')}")
    
    # Save results
    with open("gpu_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  📄 Full results saved to: gpu_test_results.json")
    
    print("\n" + "="*60)
    print("  Test completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
