"""
Pytest configuration and shared fixtures for GPU project tests.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


# ============================================================================
# Device Detection Fixtures
# ============================================================================

@pytest.fixture
def cpu_device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def available_device():
    """Return the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def has_gpu():
    """Check if any GPU is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


@pytest.fixture
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def has_mps():
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()


# ============================================================================
# Skip Markers
# ============================================================================

# Skip if no GPU available
requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="Test requires GPU (CUDA or MPS)"
)

# Skip if no CUDA available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires NVIDIA CUDA GPU"
)

# Skip if no MPS available
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Test requires Apple Silicon MPS"
)


# ============================================================================
# Mock Fixtures for No-GPU Environments
# ============================================================================

@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available for testing without actual GPU."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_name', return_value='Mock NVIDIA GPU'), \
         patch('torch.version.cuda', '12.0'), \
         patch('torch.cuda.get_device_properties') as mock_props:
        mock_props.return_value = MagicMock(total_memory=8e9)
        yield


@pytest.fixture
def mock_mps_available():
    """Mock MPS as available for testing without actual Apple Silicon."""
    with patch('torch.backends.mps.is_available', return_value=True), \
         patch('torch.backends.mps.is_built', return_value=True):
        yield


@pytest.fixture
def mock_no_gpu():
    """Mock environment with no GPU available."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        yield


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_matrix_small():
    """Small matrix for quick tests."""
    return torch.randn(64, 64)


@pytest.fixture
def sample_matrix_medium():
    """Medium matrix for moderate tests."""
    return torch.randn(512, 512)


@pytest.fixture
def sample_batch_data():
    """Sample batch data for neural network tests."""
    batch_size = 16
    input_size = 1024
    num_classes = 10
    X = torch.randn(batch_size, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    return X, y


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require NVIDIA CUDA"
    )
    config.addinivalue_line(
        "markers", "mps: marks tests that require Apple Silicon MPS"
    )


def pytest_collection_modifyitems(config, items):
    """Add skip markers based on hardware availability."""
    for item in items:
        # Auto-skip GPU tests if no GPU available
        if "gpu" in item.keywords:
            if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
                item.add_marker(pytest.mark.skip(reason="No GPU available"))

        if "cuda" in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        if "mps" in item.keywords:
            if not torch.backends.mps.is_available():
                item.add_marker(pytest.mark.skip(reason="MPS not available"))
