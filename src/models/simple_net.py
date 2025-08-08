"""
Simple neural network example for GPU demonstration.
"""

import torch
import torch.nn as nn
import time
from typing import Tuple


class SimpleNet(nn.Module):
    """A simple neural network for demonstration purposes."""
    
    def __init__(self, input_size: int = 1024, hidden_size: int = 512, output_size: int = 10):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)


def benchmark_training(
    device: torch.device,
    batch_size: int = 32,
    input_size: int = 1024,
    num_epochs: int = 10
) -> Tuple[float, float]:
    """
    Benchmark neural network training on CPU vs GPU.
    
    Args:
        device: Device to run training on
        batch_size: Batch size for training
        input_size: Input feature size
        num_epochs: Number of training epochs
    
    Returns:
        Tuple of (cpu_time, gpu_time) in seconds
    """
    # Create model and data
    model = SimpleNet(input_size=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate random data
    X = torch.randn(batch_size * 10, input_size)
    y = torch.randint(0, 10, (batch_size * 10,))
    
    # CPU benchmark
    print(f"🔄 Training neural network on CPU...")
    model_cpu = model.to('cpu')
    X_cpu, y_cpu = X.to('cpu'), y.to('cpu')
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.001)
    
    start_time = time.time()
    model_cpu.train()
    for epoch in range(num_epochs):
        for i in range(0, len(X_cpu), batch_size):
            batch_X = X_cpu[i:i+batch_size]
            batch_y = y_cpu[i:i+batch_size]
            
            optimizer_cpu.zero_grad()
            outputs = model_cpu(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_cpu.step()
    
    cpu_time = time.time() - start_time
    print(f"🖥️  CPU training time: {cpu_time:.4f} seconds")
    
    # GPU benchmark
    gpu_time = float('inf')
    if device.type in ['cuda', 'mps']:
        print(f"🔄 Training neural network on {device.type.upper()}...")
        model_gpu = SimpleNet(input_size=input_size).to(device)
        X_gpu, y_gpu = X.to(device), y.to(device)
        optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.001)
        
        # Warm up
        _ = model_gpu(X_gpu[:batch_size])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        model_gpu.train()
        for epoch in range(num_epochs):
            for i in range(0, len(X_gpu), batch_size):
                batch_X = X_gpu[i:i+batch_size]
                batch_y = y_gpu[i:i+batch_size]
                
                optimizer_gpu.zero_grad()
                outputs = model_gpu(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer_gpu.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        gpu_time = time.time() - start_time
        gpu_name = "CUDA GPU" if device.type == 'cuda' else "Apple Silicon GPU"
        print(f"🚀 {gpu_name} training time: {gpu_time:.4f} seconds")
        print(f"⚡ Training speedup: {cpu_time / gpu_time:.2f}x")
    
    return cpu_time, gpu_time


def demo_inference(device: torch.device, batch_size: int = 64):
    """Demonstrate GPU inference speed."""
    print(f"\n🧠 Neural Network Inference Demo")
    print(f"Device: {device}")
    
    model = SimpleNet().to(device)
    model.eval()
    
    # Generate test data
    test_data = torch.randn(batch_size, 1024).to(device)
    
    # Warm up
    with torch.no_grad():
        _ = model(test_data)
    
    # Benchmark inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 100 inference runs
            predictions = model(test_data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    inference_time = time.time() - start_time
    
    print(f"🏃‍♂️ 100 inference runs: {inference_time:.4f} seconds")
    print(f"📊 Average per batch: {inference_time/100*1000:.2f} ms")
    print(f"🎯 Predictions shape: {predictions.shape}")
    
    return inference_time


if __name__ == "__main__":
    from ..gpu_utils import get_device
    
    device = get_device()
    
    print("🧠 Neural Network GPU Demo")
    print("=" * 50)
    
    # Training benchmark
    benchmark_training(device, batch_size=32, num_epochs=5)
    
    # Inference demo
    demo_inference(device)
