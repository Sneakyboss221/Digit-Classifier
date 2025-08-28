"""Tests for ML models."""

import pytest
import torch
import torch.nn as nn

from models.mlp_model import MLPModel, create_mlp_model
from models.lenet5_model import LeNet5Model, create_lenet5_model
from models.resnet_model import ResNetModel, create_resnet_model


class TestMLPModel:
    """Test cases for MLPModel."""

    def test_mlp_model_creation(self):
        """Test MLP model creation."""
        model = create_mlp_model()
        assert isinstance(model, MLPModel)
        assert model is not None

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = create_mlp_model()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 784)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_mlp_with_image_input(self):
        """Test MLP with image input (should flatten automatically)."""
        model = create_mlp_model()
        batch_size = 16
        input_tensor = torch.randn(batch_size, 1, 28, 28)  # Image format

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestLeNet5Model:
    """Test cases for LeNet5Model."""

    def test_lenet5_model_creation(self):
        """Test LeNet-5 model creation."""
        model = create_lenet5_model()
        assert isinstance(model, LeNet5Model)
        assert model is not None

    def test_lenet5_forward_pass(self):
        """Test LeNet-5 forward pass."""
        model = create_lenet5_model()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestResNetModel:
    """Test cases for ResNetModel."""

    def test_resnet_model_creation(self):
        """Test ResNet model creation."""
        model = create_resnet_model()
        assert isinstance(model, ResNetModel)
        assert model is not None

    def test_resnet_forward_pass(self):
        """Test ResNet forward pass."""
        model = create_resnet_model()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestModelConsistency:
    """Test model consistency across different inputs."""

    def test_all_models_same_output_shape(self):
        """Test that all models produce the same output shape."""
        models = [create_mlp_model(), create_lenet5_model(), create_resnet_model()]

        batch_size = 16
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        for model in models:
            output = model(input_tensor)
            assert output.shape == (batch_size, 10)

    def test_model_output_probabilities(self):
        """Test that model outputs are valid probabilities."""
        models = [create_mlp_model(), create_lenet5_model(), create_resnet_model()]

        batch_size = 8
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        for model in models:
            output = model(input_tensor)
            # Check that outputs sum to 1 (log probabilities)
            log_probs = output
            probs = torch.exp(log_probs)
            assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
