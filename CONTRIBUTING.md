# 🤝 Contributing to Ultimate MNIST Digit Recognition

Thank you for your interest in contributing to the Ultimate MNIST Digit Recognition project! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs

- Use the GitHub issue tracker
- Include a clear and descriptive title
- Provide detailed steps to reproduce the bug
- Include system information (OS, Python version, etc.)
- Add screenshots if applicable

### 💡 Suggesting Enhancements

- Use the GitHub issue tracker
- Describe the enhancement clearly
- Explain why this enhancement would be useful
- Include mockups or examples if applicable

### 🔧 Code Contributions

- Fork the repository
- Create a feature branch
- Make your changes
- Add tests if applicable
- Ensure all tests pass
- Submit a pull request

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/Digit-Classifier.git
   cd Digit-Classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests**
   ```bash
   pytest
   ```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Docstrings**: Google style
- **Type hints**: Required for all functions
- **Imports**: Organized and sorted

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black .
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
flake8 .
```

### Type Checking

We use [mypy](http://mypy-lang.org/) for type checking:

```bash
mypy .
```

### Example Code Style

```python
"""Module docstring."""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    """Example model class with proper documentation.
    
    This class demonstrates the coding standards we follow.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_classes: Number of output classes
        
    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        dropout: Dropout layer for regularization
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        """Initialize the model."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def example_function(data: np.ndarray, threshold: float = 0.5) -> Tuple[bool, Optional[str]]:
    """Example function with type hints and documentation.
    
    Args:
        data: Input data array
        threshold: Threshold value for processing
        
    Returns:
        Tuple containing success status and optional message
    """
    if data.size == 0:
        return False, "Empty data provided"
    
    result = np.mean(data) > threshold
    return result, None
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Tests should be in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Use fixtures for common setup

### Example Test

```python
"""Test module for models."""

import pytest
import torch
from models.mlp_model import MLPModel


class TestMLPModel:
    """Test cases for MLPModel."""
    
    @pytest.fixture
    def model(self) -> MLPModel:
        """Create a test model instance."""
        return MLPModel(input_size=784, hidden_size=128, num_classes=10)
    
    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Create sample input tensor."""
        return torch.randn(32, 784)
    
    def test_model_initialization(self, model: MLPModel) -> None:
        """Test model initialization."""
        assert model is not None
        assert isinstance(model, MLPModel)
    
    def test_forward_pass(self, model: MLPModel, sample_input: torch.Tensor) -> None:
        """Test forward pass through the model."""
        output = model(sample_input)
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_invalid_input_size(self) -> None:
        """Test model with invalid input size."""
        with pytest.raises(ValueError):
            MLPModel(input_size=-1, hidden_size=128, num_classes=10)
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   pytest
   ```

2. **Check code formatting**
   ```bash
   black .
   flake8 .
   mypy .
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what and why, not how
3. **Related issues**: Link to relevant issues
4. **Screenshots**: Include for UI changes
5. **Tests**: Ensure all tests pass

### Example Pull Request

```markdown
## Description
Add support for VGG model architecture to improve digit recognition accuracy.

## Changes
- Add VGG model implementation in `models/vgg_model.py`
- Update ensemble model to include VGG
- Add VGG configuration to `config.py`
- Include VGG in training pipeline

## Testing
- Added unit tests for VGG model
- All existing tests pass
- Manual testing with web interface

## Related Issues
Closes #123

## Screenshots
[Include screenshots if UI changes]
```

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12.0]
- Python Version: [e.g., 3.9.7]
- Package Versions: [e.g., torch 1.12.0, streamlit 1.22.0]

## Additional Information
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Brief description of the feature

## Use Case
Why this feature would be useful

## Proposed Implementation
How you think it should be implemented

## Alternatives Considered
Other approaches you considered

## Additional Information
Any other relevant information
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Ultimate MNIST Digit Recognition! 🚀
