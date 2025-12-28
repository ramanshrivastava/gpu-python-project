#!/usr/bin/env python3
"""
Apple Silicon GPU Analysis and Capability Detection
Comprehensive script to analyze Apple Silicon GPU capabilities and performance.
"""

import subprocess
import sys
import torch
import platform
import psutil
import time
import numpy as np
from typing import Dict, Any


def run_system_command(command: str) -> str:
    """Run a system command and return output."""
    try:
        result = subprocess.run(command.split(), capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_apple_silicon_specs() -> Dict[str, Any]:
    """Get detailed Apple Silicon specifications."""
    specs = {}
    
    # Basic hardware info
    hw_info = run_system_command("system_profiler SPHardwareDataType")
    gpu_info = run_system_command("system_profiler SPDisplaysDataType")
    
    # Parse chip model
    for line in hw_info.split('\n'):
        if 'Chip:' in line:
            specs['chip_model'] = line.split('Chip:')[1].strip()
        elif 'Total Number of Cores:' in line:
            specs['cpu_cores'] = line.split('Total Number of Cores:')[1].strip()
        elif 'Memory:' in line:
            specs['unified_memory'] = line.split('Memory:')[1].strip()
    
    # Parse GPU cores
    for line in gpu_info.split('\n'):
        if 'Total Number of Cores:' in line and 'GPU' in gpu_info:
            specs['gpu_cores'] = line.split('Total Number of Cores:')[1].strip()
        elif 'Metal Support:' in line:
            specs['metal_version'] = line.split('Metal Support:')[1].strip()
    
    return specs


def get_memory_details() -> Dict[str, Any]:
    """Get detailed memory information."""
    memory_info = {}
    
    # System memory
    memory_info['total_memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    memory_info['available_memory_gb'] = round(psutil.virtual_memory().available / (1024**3), 2)
    
    # Unified memory architecture
    memory_info['unified_memory'] = True  # Apple Silicon uses unified memory
    memory_info['memory_bandwidth'] = get_memory_bandwidth_estimate()
    
    return memory_info


def get_memory_bandwidth_estimate() -> str:
    """Estimate memory bandwidth based on chip model."""
    try:
        chip_info = run_system_command("system_profiler SPHardwareDataType")
        if "M1 Pro" in chip_info:
            return "~200 GB/s (estimated for M1 Pro)"
        elif "M1 Max" in chip_info:
            return "~400 GB/s (estimated for M1 Max)"
        elif "M2 Pro" in chip_info:
            return "~200 GB/s (estimated for M2 Pro)"
        elif "M2 Max" in chip_info:
            return "~400 GB/s (estimated for M2 Max)"
        elif "M3" in chip_info:
            return "~100-400 GB/s (estimated for M3 series)"
        else:
            return "Unknown"
    except:
        return "Unable to estimate"


def analyze_pytorch_capabilities() -> Dict[str, Any]:
    """Analyze PyTorch GPU capabilities."""
    pytorch_info = {}
    
    # Basic PyTorch info
    pytorch_info['pytorch_version'] = torch.__version__
    pytorch_info['mps_available'] = torch.backends.mps.is_available()
    pytorch_info['mps_built'] = torch.backends.mps.is_built()
    
    if pytorch_info['mps_available']:
        # Test device creation
        try:
            device = torch.device('mps')
            test_tensor = torch.randn(100, 100).to(device)
            pytorch_info['mps_functional'] = True
            pytorch_info['device_name'] = 'Apple Silicon GPU (MPS)'
        except Exception as e:
            pytorch_info['mps_functional'] = False
            pytorch_info['error'] = str(e)
    
    return pytorch_info


def benchmark_compute_capabilities() -> Dict[str, Any]:
    """Benchmark compute capabilities."""
    print("🔄 Running compute capability benchmarks...")
    
    benchmarks = {}
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    
    # Matrix multiplication benchmark
    sizes = [512, 1024, 2048]
    matrix_results = {}
    
    for size in sizes:
        print(f"  Testing {size}x{size} matrices...")
        
        # CPU benchmark
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start_time = time.time()
        _ = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        # GPU benchmark (if available)
        if device.type == 'mps':
            a_gpu = a_cpu.to(device)
            b_gpu = b_cpu.to(device)
            
            # Warm up
            _ = torch.mm(a_gpu, b_gpu)
            torch.mps.synchronize()
            
            start_time = time.time()
            _ = torch.mm(a_gpu, b_gpu)
            torch.mps.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            matrix_results[f"{size}x{size}"] = {
                'cpu_time': round(cpu_time, 6),
                'gpu_time': round(gpu_time, 6), 
                'speedup': round(speedup, 2)
            }
        else:
            matrix_results[f"{size}x{size}"] = {
                'cpu_time': round(cpu_time, 6),
                'gpu_time': 'N/A',
                'speedup': 'N/A'
            }
    
    benchmarks['matrix_multiplication'] = matrix_results
    
    # Neural network operations
    if device.type == 'mps':
        benchmarks['neural_ops'] = benchmark_neural_operations(device)
    
    return benchmarks


def benchmark_neural_operations(device: torch.device) -> Dict[str, Any]:
    """Benchmark common neural network operations."""
    print("  Testing neural network operations...")
    
    neural_benchmarks = {}
    
    # Convolution benchmark
    try:
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        conv_layer = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        # Warm up
        _ = conv_layer(input_tensor)
        torch.mps.synchronize()
        
        start_time = time.time()
        for _ in range(10):
            _ = conv_layer(input_tensor)
        torch.mps.synchronize()
        conv_time = time.time() - start_time
        
        neural_benchmarks['convolution_10x'] = round(conv_time, 6)
    except Exception as e:
        neural_benchmarks['convolution_error'] = str(e)
    
    # Linear layer benchmark
    try:
        input_tensor = torch.randn(64, 1024).to(device)
        linear_layer = torch.nn.Linear(1024, 512).to(device)
        
        # Warm up
        _ = linear_layer(input_tensor)
        torch.mps.synchronize()
        
        start_time = time.time()
        for _ in range(100):
            _ = linear_layer(input_tensor)
        torch.mps.synchronize()
        linear_time = time.time() - start_time
        
        neural_benchmarks['linear_100x'] = round(linear_time, 6)
    except Exception as e:
        neural_benchmarks['linear_error'] = str(e)
    
    return neural_benchmarks


def get_gpu_capabilities() -> Dict[str, Any]:
    """Get Apple Silicon GPU specific capabilities."""
    capabilities = {}
    
    # Metal capabilities
    if torch.backends.mps.is_available():
        capabilities['metal_performance_shaders'] = True
        capabilities['unified_memory_architecture'] = True
        capabilities['shared_memory_with_cpu'] = True
        
        # Supported data types
        device = torch.device('mps')
        supported_dtypes = []
        
        test_dtypes = [
            ('float32', torch.float32),
            ('float16', torch.float16),
            ('int32', torch.int32),
            ('int64', torch.int64),
            ('bool', torch.bool)
        ]
        
        for name, dtype in test_dtypes:
            try:
                test_tensor = torch.tensor([1.0], dtype=dtype).to(device)
                supported_dtypes.append(name)
            except:
                pass
        
        capabilities['supported_dtypes'] = supported_dtypes
        
        # GPU-specific features
        capabilities['features'] = [
            'Automatic memory management',
            'Zero-copy CPU-GPU transfers',
            'Hardware-accelerated matrix operations',
            'Optimized neural network primitives',
            'Metal Performance Shaders integration',
            'Power-efficient compute',
            'Shared memory pool with CPU'
        ]
    
    return capabilities


def main():
    """Main function to run comprehensive Apple Silicon GPU analysis."""
    print("🍎 Apple Silicon GPU Comprehensive Analysis")
    print("=" * 70)
    
    # System specifications
    print("\n📋 System Specifications:")
    specs = get_apple_silicon_specs()
    for key, value in specs.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Memory details
    print("\n💾 Memory Architecture:")
    memory_info = get_memory_details()
    for key, value in memory_info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # PyTorch capabilities
    print("\n🔥 PyTorch Integration:")
    pytorch_info = analyze_pytorch_capabilities()
    for key, value in pytorch_info.items():
        if key != 'error':
            print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"  Error: {value}")
    
    # GPU capabilities
    print("\n⚡ GPU Capabilities:")
    capabilities = get_gpu_capabilities()
    if 'features' in capabilities:
        for feature in capabilities['features']:
            print(f"  ✅ {feature}")
        print(f"  📊 Supported Data Types: {', '.join(capabilities['supported_dtypes'])}")
    
    # Performance benchmarks
    print(f"\n🚀 Performance Benchmarks:")
    benchmarks = benchmark_compute_capabilities()
    
    if 'matrix_multiplication' in benchmarks:
        print("  Matrix Multiplication Results:")
        for size, results in benchmarks['matrix_multiplication'].items():
            cpu_time = results['cpu_time']
            gpu_time = results['gpu_time']
            speedup = results['speedup']
            
            if gpu_time != 'N/A':
                print(f"    {size}: CPU={cpu_time}s, GPU={gpu_time}s, Speedup={speedup}x")
            else:
                print(f"    {size}: CPU={cpu_time}s (GPU not available)")
    
    if 'neural_ops' in benchmarks:
        print("  Neural Network Operations:")
        for op, timing in benchmarks['neural_ops'].items():
            print(f"    {op.replace('_', ' ').title()}: {timing}s")
    
    # Recommendations
    print(f"\n💡 Optimization Recommendations:")
    print("  🔸 Use batch sizes ≥32 for optimal GPU utilization")
    print("  🔸 Keep tensors on GPU between operations to avoid transfers")
    print("  🔸 Use float16 for memory-intensive workloads when possible")
    print("  🔸 Leverage unified memory for seamless CPU-GPU data sharing")
    print("  🔸 Prefer larger matrix operations (>1024×1024) for GPU acceleration")
    
    print(f"\n✅ Analysis completed successfully!")


if __name__ == "__main__":
    main()
