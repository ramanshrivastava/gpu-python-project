# GPU Python Project 🚀

A comprehensive Python project demonstrating GPU computing with PyTorch, supporting both **NVIDIA CUDA** and **Apple Silicon MPS** backends.

## ✨ Features
- **Multi-GPU Backend Support**: CUDA (NVIDIA) and MPS (Apple Silicon)
- **Performance Benchmarking**: Matrix operations and neural network training
- **Automatic Device Detection**: Intelligently selects best available compute device
- **Memory Management**: GPU memory monitoring and cleanup utilities
- **Neural Network Examples**: Complete training and inference pipelines
- **Robust Error Handling**: Graceful fallback to CPU when GPU unavailable

## 🏆 Performance Results

Tested on **Apple M1 Pro** with MPS backend:

### Matrix Multiplication Benchmarks
| Matrix Size | CPU Time | GPU Time | Speedup | Winner |
|-------------|----------|----------|---------|--------|
| 512×512     | 0.0004s  | 0.0006s  | 0.66x   | CPU ⚡ |
| 1024×1024   | 0.0027s  | 0.0021s  | 1.30x   | GPU 🚀 |
| 2048×2048   | 0.0180s  | 0.0082s  | 2.19x   | GPU 🚀 |
| 4096×4096   | 0.1033s  | 0.0318s  | **3.25x** | GPU 🚀 |

### Neural Network Performance
- **Inference Speed**: 0.33ms per batch (64 samples)
- **Training**: Variable performance depending on model size
- **Best Use Cases**: Large models, batch inference, parallel computations

## 🚀 Quick Start

### Local Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd gpu-python-project

# Create and activate virtual environment
python -m venv gpu-env
source gpu-env/bin/activate  # On Windows: gpu-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage Examples

### 1. Basic GPU Detection and Info
```bash
# Test GPU availability and show system info
python src/gpu_utils.py
```

### 2. Performance Benchmarking
```bash
# Run comprehensive matrix multiplication benchmarks
python -c "
from src.gpu_utils import get_device, benchmark_matrix_multiplication
device = get_device()
for size in [512, 1024, 2048, 4096]:
    print(f'\n--- {size}x{size} Matrix ---')
    benchmark_matrix_multiplication(size, device)
"
```

### 3. Neural Network Training and Inference
```bash
# Test neural network on GPU
python -c "
import sys
sys.path.append('src')
from gpu_utils import get_device
from models.simple_net import benchmark_training, demo_inference

device = get_device()
benchmark_training(device, batch_size=32, num_epochs=3)
demo_inference(device, batch_size=64)
"
```

### 4. Custom GPU Operations
```python
from src.gpu_utils import get_device, memory_info, clear_gpu_memory
import torch

# Get optimal device
device = get_device()

# Create tensors on GPU
a = torch.randn(1000, 1000).to(device)
b = torch.randn(1000, 1000).to(device)

# Perform GPU computation
result = torch.mm(a, b)

# Check memory usage (CUDA only)
if device.type == 'cuda':
    print(memory_info())
    clear_gpu_memory()
```

## 🖥️ Supported Platforms

### ✅ Apple Silicon (M1/M2/M3)
- **Backend**: Metal Performance Shaders (MPS)
- **Installation**: Standard PyTorch (MPS included)
- **Performance**: Excellent for matrix operations and inference

### ✅ NVIDIA GPUs
- **Backend**: CUDA
- **Installation**: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **Requirements**: CUDA drivers and toolkit

### ✅ CPU Fallback
- **Backend**: Standard PyTorch CPU
- **Performance**: Reliable baseline for all operations

## 🛠️ Development

### Project Structure
```
gpu-python-project/
├── src/
│   ├── __init__.py
│   ├── gpu_utils.py          # GPU utilities and benchmarking
│   └── models/
│       ├── __init__.py
│       └── simple_net.py     # Neural network examples
├── tests/                    # Unit tests (coming soon)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

### Adding New GPU Operations
1. Extend `gpu_utils.py` with new benchmark functions
2. Add device synchronization for accurate timing
3. Include both CPU and GPU implementations for comparison
4. Handle multiple backend types (`cuda`, `mps`, `cpu`)

## 📋 Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+ (with MPS support)
- **NumPy**: 1.24+
- **Hardware**: 
  - Apple Silicon Mac (for MPS)
  - NVIDIA GPU with CUDA support (for CUDA)
  - Any CPU (fallback)

## 🚀 Performance Tips

1. **GPU Excels At**:
   - Large matrix operations (>1024×1024)
   - Batch processing
   - Parallel computations
   - Deep learning inference

2. **CPU Better For**:
   - Small operations
   - Sequential tasks
   - Memory-limited scenarios
   - Simple computations with overhead

3. **Optimization Strategies**:
   - Use larger batch sizes on GPU
   - Keep data on GPU between operations
   - Profile memory usage to avoid bottlenecks
   - Consider mixed precision for large models

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🔗 Useful Links

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [Apple Silicon ML Performance](https://developer.apple.com/metal/pytorch/)
