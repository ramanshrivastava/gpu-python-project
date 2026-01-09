"""
Tests for gpu_utils module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpu_utils import (
    get_device,
    benchmark_matrix_multiplication,
    memory_info,
    clear_gpu_memory
)
from conftest import requires_gpu, requires_cuda, requires_mps


# ============================================================================
# get_device() Tests
# ============================================================================

class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_torch_device(self):
        """get_device should return a torch.device object."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_is_valid(self):
        """Device type should be one of cuda, mps, or cpu."""
        device = get_device()
        assert device.type in ['cuda', 'mps', 'cpu']

    def test_returns_cuda_when_available(self):
        """Should return CUDA device when CUDA is available."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=True), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=False), \
             patch('gpu_utils.torch.cuda.get_device_name', return_value='Mock GPU'), \
             patch('gpu_utils.torch.version.cuda', '12.0'), \
             patch('gpu_utils.torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value = MagicMock(total_memory=8e9)
            device = get_device()
            assert device.type == 'cuda'

    def test_returns_mps_when_cuda_unavailable(self):
        """Should return MPS device when CUDA unavailable but MPS is."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=False), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=True), \
             patch('gpu_utils.torch.backends.mps.is_built', return_value=True):
            device = get_device()
            assert device.type == 'mps'

    def test_returns_cpu_when_no_gpu(self):
        """Should return CPU device when no GPU available."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=False), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=False):
            device = get_device()
            assert device.type == 'cpu'

    def test_device_is_usable(self, available_device):
        """Returned device should be usable for tensor operations."""
        device = get_device()
        tensor = torch.randn(10, 10).to(device)
        assert tensor.device.type == device.type


# ============================================================================
# benchmark_matrix_multiplication() Tests
# ============================================================================

class TestBenchmarkMatrixMultiplication:
    """Tests for benchmark_matrix_multiplication function."""

    @pytest.mark.parametrize("matrix_size", [64, 128, 256])
    def test_returns_tuple_of_floats(self, matrix_size, cpu_device):
        """Should return tuple of (cpu_time, gpu_time)."""
        cpu_time, gpu_time = benchmark_matrix_multiplication(matrix_size, cpu_device)
        assert isinstance(cpu_time, float)
        assert isinstance(gpu_time, float)

    @pytest.mark.parametrize("matrix_size", [64, 128, 256])
    def test_cpu_time_is_positive(self, matrix_size, cpu_device):
        """CPU time should be positive."""
        cpu_time, _ = benchmark_matrix_multiplication(matrix_size, cpu_device)
        assert cpu_time > 0

    def test_gpu_time_is_inf_on_cpu(self, cpu_device):
        """GPU time should be inf when running on CPU only."""
        _, gpu_time = benchmark_matrix_multiplication(128, cpu_device)
        assert gpu_time == float('inf')

    @requires_gpu
    @pytest.mark.parametrize("matrix_size", [256, 512])
    def test_gpu_time_is_positive_with_gpu(self, matrix_size, available_device):
        """GPU time should be positive when GPU is available."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")
        _, gpu_time = benchmark_matrix_multiplication(matrix_size, available_device)
        assert gpu_time > 0
        assert gpu_time != float('inf')

    @requires_gpu
    def test_results_are_consistent(self, available_device):
        """Multiple runs should give similar timing results."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        times = []
        for _ in range(3):
            _, gpu_time = benchmark_matrix_multiplication(256, available_device)
            times.append(gpu_time)

        # Check variance is reasonable (within 5x of each other)
        assert max(times) / min(times) < 5

    def test_default_device_selection(self):
        """Should select device automatically if not provided."""
        cpu_time, gpu_time = benchmark_matrix_multiplication(64)
        assert cpu_time > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("matrix_size", [512, 1024])
    def test_larger_matrices(self, matrix_size, cpu_device):
        """Test with larger matrices (slower)."""
        cpu_time, _ = benchmark_matrix_multiplication(matrix_size, cpu_device)
        assert cpu_time > 0


# ============================================================================
# memory_info() Tests
# ============================================================================

class TestMemoryInfo:
    """Tests for memory_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        info = memory_info()
        assert isinstance(info, dict)

    def test_has_backend_or_error_key(self):
        """Should have either 'backend' or 'error' key."""
        info = memory_info()
        assert 'backend' in info or 'error' in info

    @requires_cuda
    def test_cuda_memory_info_has_expected_keys(self):
        """CUDA memory info should have specific keys."""
        info = memory_info()
        if info.get('backend') == 'CUDA':
            expected_keys = ['backend', 'allocated_gb', 'cached_gb', 'max_allocated_gb', 'total_gb']
            for key in expected_keys:
                assert key in info

    @requires_cuda
    def test_cuda_memory_values_are_non_negative(self):
        """CUDA memory values should be non-negative."""
        info = memory_info()
        if info.get('backend') == 'CUDA':
            assert info['allocated_gb'] >= 0
            assert info['cached_gb'] >= 0
            assert info['total_gb'] > 0

    @requires_mps
    def test_mps_memory_info_has_status(self):
        """MPS memory info should indicate limited tracking."""
        info = memory_info()
        if info.get('backend') == 'MPS (Apple Silicon)':
            assert 'status' in info

    def test_no_gpu_returns_error(self):
        """Should return error when no GPU available."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=False), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=False):
            info = memory_info()
            assert 'error' in info

    def test_mock_cuda_returns_cuda_info(self):
        """Mock CUDA should return CUDA memory info structure."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=True), \
             patch('gpu_utils.torch.cuda.memory_allocated', return_value=1e9), \
             patch('gpu_utils.torch.cuda.memory_reserved', return_value=2e9), \
             patch('gpu_utils.torch.cuda.max_memory_allocated', return_value=1.5e9), \
             patch('gpu_utils.torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value = MagicMock(total_memory=8e9)
            info = memory_info()
            assert info['backend'] == 'CUDA'
            assert info['allocated_gb'] == pytest.approx(1.0, rel=0.01)


# ============================================================================
# clear_gpu_memory() Tests
# ============================================================================

class TestClearGpuMemory:
    """Tests for clear_gpu_memory function."""

    def test_does_not_raise_on_cpu(self):
        """Should not raise exception when no GPU available."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=False), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=False):
            # Should not raise
            clear_gpu_memory()

    @requires_cuda
    def test_clears_cuda_cache(self):
        """Should call torch.cuda.empty_cache() on CUDA."""
        with patch('gpu_utils.torch.cuda.empty_cache') as mock_empty:
            with patch('gpu_utils.torch.cuda.is_available', return_value=True):
                clear_gpu_memory()
                mock_empty.assert_called_once()

    def test_mock_cuda_calls_empty_cache(self):
        """Mock CUDA clear should call empty_cache."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=True), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=False), \
             patch('gpu_utils.torch.cuda.empty_cache') as mock_empty:
            clear_gpu_memory()
            mock_empty.assert_called_once()

    def test_mock_mps_does_not_raise(self):
        """MPS clear should not raise (no explicit clear available)."""
        with patch('gpu_utils.torch.cuda.is_available', return_value=False), \
             patch('gpu_utils.torch.backends.mps.is_available', return_value=True):
            # Should not raise
            clear_gpu_memory()


# ============================================================================
# Integration Tests
# ============================================================================

class TestGpuUtilsIntegration:
    """Integration tests for gpu_utils module."""

    def test_full_workflow_on_cpu(self, cpu_device):
        """Test complete workflow on CPU."""
        # Get device
        device = get_device()

        # Run benchmark
        cpu_time, gpu_time = benchmark_matrix_multiplication(64, cpu_device)
        assert cpu_time > 0

        # Check memory
        info = memory_info()
        assert isinstance(info, dict)

        # Clear memory (should not fail)
        clear_gpu_memory()

    @requires_gpu
    def test_full_workflow_on_gpu(self, available_device):
        """Test complete workflow on GPU."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        # Run benchmark
        cpu_time, gpu_time = benchmark_matrix_multiplication(256, available_device)
        assert cpu_time > 0
        assert gpu_time > 0
        assert gpu_time != float('inf')

        # Check memory
        info = memory_info()
        assert 'backend' in info

        # Clear memory
        clear_gpu_memory()

    @pytest.mark.parametrize("size", [32, 64, 128])
    def test_benchmark_various_sizes(self, size, cpu_device):
        """Test benchmarks with various matrix sizes."""
        cpu_time, _ = benchmark_matrix_multiplication(size, cpu_device)
        assert cpu_time > 0
