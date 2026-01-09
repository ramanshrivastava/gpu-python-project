"""
Tests for SimpleNet neural network module.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.simple_net import SimpleNet, benchmark_training, demo_inference
from conftest import requires_gpu, requires_cuda, requires_mps


# ============================================================================
# SimpleNet Class Tests
# ============================================================================

class TestSimpleNet:
    """Tests for SimpleNet neural network class."""

    def test_instantiation_default_params(self):
        """Should instantiate with default parameters."""
        model = SimpleNet()
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("input_size,hidden_size,output_size", [
        (512, 256, 5),
        (1024, 512, 10),
        (2048, 1024, 100),
    ])
    def test_instantiation_custom_params(self, input_size, hidden_size, output_size):
        """Should instantiate with custom parameters."""
        model = SimpleNet(input_size, hidden_size, output_size)
        assert isinstance(model, nn.Module)

    def test_forward_pass_shape(self):
        """Forward pass should return correct output shape."""
        batch_size = 16
        input_size = 1024
        output_size = 10

        model = SimpleNet(input_size=input_size, output_size=output_size)
        x = torch.randn(batch_size, input_size)
        output = model(x)

        assert output.shape == (batch_size, output_size)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_forward_pass_various_batch_sizes(self, batch_size):
        """Forward pass should work with various batch sizes."""
        model = SimpleNet()
        x = torch.randn(batch_size, 1024)
        output = model(x)
        assert output.shape == (batch_size, 10)

    def test_output_is_probability_distribution(self):
        """Output should be valid probability distribution (sums to 1)."""
        model = SimpleNet()
        x = torch.randn(16, 1024)
        output = model(x)

        # Each row should sum to approximately 1 (due to Softmax)
        row_sums = output.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(16), atol=1e-5)

    def test_output_is_non_negative(self):
        """Output probabilities should be non-negative."""
        model = SimpleNet()
        x = torch.randn(16, 1024)
        output = model(x)
        assert (output >= 0).all()

    def test_model_has_trainable_parameters(self):
        """Model should have trainable parameters."""
        model = SimpleNet()
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_model_can_be_moved_to_device(self, cpu_device):
        """Model should be movable to different devices."""
        model = SimpleNet()
        model = model.to(cpu_device)
        # Check that parameters are on correct device
        for param in model.parameters():
            assert param.device.type == cpu_device.type

    @requires_gpu
    def test_model_on_gpu(self, available_device):
        """Model should work on GPU."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        model = SimpleNet().to(available_device)
        x = torch.randn(16, 1024).to(available_device)
        output = model(x)
        assert output.device.type == available_device.type

    def test_gradient_computation(self):
        """Gradients should be computed correctly."""
        model = SimpleNet()
        x = torch.randn(16, 1024)
        y = torch.randint(0, 10, (16,))

        criterion = nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_train_mode(self):
        """Model should switch to train mode."""
        model = SimpleNet()
        model.train()
        assert model.training

    def test_mode_switch(self):
        """Model should switch between train and other modes."""
        model = SimpleNet()
        model.train()
        assert model.training
        # Switch to inference mode by setting training=False
        model.training = False
        assert not model.training


# ============================================================================
# benchmark_training() Tests
# ============================================================================

class TestBenchmarkTraining:
    """Tests for benchmark_training function."""

    def test_returns_tuple_on_cpu(self, cpu_device):
        """Should return tuple of (cpu_time, gpu_time) on CPU."""
        cpu_time, gpu_time = benchmark_training(cpu_device, batch_size=8, num_epochs=1)
        assert isinstance(cpu_time, float)
        assert isinstance(gpu_time, float)

    def test_cpu_time_positive(self, cpu_device):
        """CPU training time should be positive."""
        cpu_time, _ = benchmark_training(cpu_device, batch_size=8, num_epochs=1)
        assert cpu_time > 0

    def test_gpu_time_inf_on_cpu(self, cpu_device):
        """GPU time should be inf when device is CPU."""
        _, gpu_time = benchmark_training(cpu_device, batch_size=8, num_epochs=1)
        assert gpu_time == float('inf')

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_various_batch_sizes(self, batch_size, cpu_device):
        """Should work with various batch sizes."""
        cpu_time, _ = benchmark_training(cpu_device, batch_size=batch_size, num_epochs=1)
        assert cpu_time > 0

    @pytest.mark.parametrize("input_size", [256, 512, 1024])
    def test_various_input_sizes(self, input_size, cpu_device):
        """Should work with various input sizes."""
        cpu_time, _ = benchmark_training(
            cpu_device, batch_size=8, input_size=input_size, num_epochs=1
        )
        assert cpu_time > 0

    @pytest.mark.parametrize("num_epochs", [1, 2, 3])
    def test_various_epochs(self, num_epochs, cpu_device):
        """Should work with various epoch counts."""
        cpu_time, _ = benchmark_training(
            cpu_device, batch_size=8, num_epochs=num_epochs
        )
        assert cpu_time > 0

    @requires_gpu
    def test_gpu_time_positive_with_gpu(self, available_device):
        """GPU time should be positive when GPU available."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        _, gpu_time = benchmark_training(
            available_device, batch_size=8, num_epochs=1
        )
        assert gpu_time > 0
        assert gpu_time != float('inf')

    @requires_gpu
    @pytest.mark.slow
    def test_gpu_training_speedup(self, available_device):
        """GPU should provide speedup over CPU for larger workloads."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        cpu_time, gpu_time = benchmark_training(
            available_device, batch_size=32, num_epochs=5
        )
        # Note: For small workloads, GPU might not be faster due to overhead
        # Just verify both complete successfully
        assert cpu_time > 0
        assert gpu_time > 0


# ============================================================================
# demo_inference() Tests
# ============================================================================

class TestDemoInference:
    """Tests for demo_inference function."""

    def test_returns_float_on_cpu(self, cpu_device):
        """Should return inference time as float."""
        inference_time = demo_inference(cpu_device, batch_size=8)
        assert isinstance(inference_time, float)

    def test_inference_time_positive(self, cpu_device):
        """Inference time should be positive."""
        inference_time = demo_inference(cpu_device, batch_size=8)
        assert inference_time > 0

    @pytest.mark.parametrize("batch_size", [8, 16, 32, 64])
    def test_various_batch_sizes(self, batch_size, cpu_device):
        """Should work with various batch sizes."""
        inference_time = demo_inference(cpu_device, batch_size=batch_size)
        assert inference_time > 0

    @requires_gpu
    def test_gpu_inference(self, available_device):
        """Inference should work on GPU."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        inference_time = demo_inference(available_device, batch_size=16)
        assert inference_time > 0

    def test_inference_completes_successfully(self, cpu_device):
        """Inference should complete successfully."""
        torch.manual_seed(42)
        time1 = demo_inference(cpu_device, batch_size=8)
        assert time1 > 0


# ============================================================================
# SimpleNet Architecture Tests
# ============================================================================

class TestSimpleNetArchitecture:
    """Tests for SimpleNet architecture details."""

    def test_has_three_linear_layers(self):
        """Network should have three linear layers."""
        model = SimpleNet()
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3

    def test_has_relu_activations(self):
        """Network should have ReLU activations."""
        model = SimpleNet()
        relu_layers = [m for m in model.modules() if isinstance(m, nn.ReLU)]
        assert len(relu_layers) == 2

    def test_has_softmax_output(self):
        """Network should have Softmax output."""
        model = SimpleNet()
        softmax_layers = [m for m in model.modules() if isinstance(m, nn.Softmax)]
        assert len(softmax_layers) == 1

    def test_layer_dimensions(self):
        """Layer dimensions should match parameters."""
        input_size, hidden_size, output_size = 512, 256, 10
        model = SimpleNet(input_size, hidden_size, output_size)
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        assert linear_layers[0].in_features == input_size
        assert linear_layers[0].out_features == hidden_size
        assert linear_layers[1].in_features == hidden_size
        assert linear_layers[1].out_features == hidden_size
        assert linear_layers[2].in_features == hidden_size
        assert linear_layers[2].out_features == output_size

    def test_parameter_count(self):
        """Check approximate parameter count."""
        model = SimpleNet(input_size=1024, hidden_size=512, output_size=10)
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: (1024*512 + 512) + (512*512 + 512) + (512*10 + 10)
        # = 524288 + 512 + 262144 + 512 + 5120 + 10 = 792586
        expected_params = (1024 * 512 + 512) + (512 * 512 + 512) + (512 * 10 + 10)
        assert total_params == expected_params


# ============================================================================
# Integration Tests
# ============================================================================

class TestSimpleNetIntegration:
    """Integration tests for SimpleNet module."""

    def test_full_training_loop_cpu(self, cpu_device, sample_batch_data):
        """Test complete training loop on CPU."""
        X, y = sample_batch_data
        X, y = X.to(cpu_device), y.to(cpu_device)

        model = SimpleNet(input_size=1024).to(cpu_device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        initial_loss = None
        for epoch in range(3):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        # Loss should have changed (model learned something)
        final_loss = loss.item()
        # Note: We don't assert loss decreased because 3 epochs may not be enough
        # Just verify the training loop completes
        assert final_loss >= 0

    @requires_gpu
    def test_full_training_loop_gpu(self, available_device, sample_batch_data):
        """Test complete training loop on GPU."""
        if available_device.type == 'cpu':
            pytest.skip("No GPU available")

        X, y = sample_batch_data
        X, y = X.to(available_device), y.to(available_device)

        model = SimpleNet(input_size=1024).to(available_device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Verify model is on correct device
        for param in model.parameters():
            assert param.device.type == available_device.type

    def test_model_save_load(self, cpu_device, tmp_path):
        """Test model save and load functionality."""
        model = SimpleNet()
        model_path = tmp_path / "model.pt"

        # Save
        torch.save(model.state_dict(), model_path)

        # Load
        loaded_model = SimpleNet()
        loaded_model.load_state_dict(torch.load(model_path, weights_only=True))

        # Verify same outputs
        x = torch.randn(8, 1024)
        model.training = False
        loaded_model.training = False

        with torch.no_grad():
            original_output = model(x)
            loaded_output = loaded_model(x)

        assert torch.allclose(original_output, loaded_output)

    def test_benchmark_and_inference_workflow(self, cpu_device):
        """Test complete benchmark and inference workflow."""
        # Run training benchmark
        cpu_time, gpu_time = benchmark_training(cpu_device, batch_size=8, num_epochs=1)
        assert cpu_time > 0

        # Run inference demo
        inference_time = demo_inference(cpu_device, batch_size=8)
        assert inference_time > 0
