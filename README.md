# 🔢 MNIST Digit Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive, production-ready MNIST digit recognition system featuring multiple machine learning models with interactive web interface, advanced visualizations, and ensemble predictions.

## 🚀 Features

### 🤖 Multiple ML Models
- **MLP (Multi-Layer Perceptron)**: Fast, lightweight neural network (~98.5% accuracy)
- **LeNet-5**: Classic CNN architecture for digit recognition (~99.2% accuracy)
- **ResNet**: Advanced residual network with skip connections (~99.4% accuracy)
- **Ensemble Model**: Combines predictions from all models (~99.6% accuracy)

### 🎨 Interactive Interface
- **Real-time drawing canvas** with instant predictions
- **Live confidence scores** and probability distributions
- **Model comparison** and selection
- **Beautiful visualizations** with Plotly charts
- **Responsive web design** with Streamlit

### 📊 Advanced Analytics
- **Training/validation curves** with real-time plotting
- **Confusion matrices** for model evaluation
- **Feature map visualization** for CNN layers
- **Misclassified digit gallery** for error analysis
- **Ensemble prediction breakdown** showing individual model contributions

### 🔧 Technical Excellence
- **Data augmentation** (rotation, shift, zoom, noise)
- **Hyperparameter tuning** capabilities
- **GPU acceleration** with automatic CUDA detection
- **Modular architecture** for easy extension
- **Production-ready** error handling and logging

## 📁 Project Structure

```
Digit-Classifier/
├── 📁 models/                    # Machine Learning Models
│   ├── mlp_model.py             # Multi-Layer Perceptron
│   ├── lenet5_model.py          # LeNet-5 CNN
│   ├── resnet_model.py          # ResNet Architecture
│   └── ensemble_model.py        # Ensemble Prediction Logic
├── 📁 data/                     # Data Processing
│   └── data_loader.py           # MNIST Data Loading & Preprocessing
├── 📁 interface/                # User Interfaces
│   ├── streamlit_app.py         # Main Web Application
│   └── drawing_canvas.py        # Pygame Drawing Canvas
├── 📁 utils/                    # Utilities
│   ├── visualization.py         # Plotting & Visualization
│   └── metrics.py               # Evaluation Metrics
├── 📁 training/                 # Training Pipeline
│   └── train_models.py          # Model Training Scripts
├── 📁 saved_models/             # Pre-trained Model Weights
├── 📁 results/                  # Training Results & Plots
├── 📄 main.py                   # Main Application Entry Point
├── 📄 quick_start.py            # Quick Setup Script
├── 📄 demo.py                   # Feature Demonstration
├── 📄 config.py                 # Configuration Management
├── 📄 requirements.txt          # Python Dependencies
└── 📄 README.md                 # Project Documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (optional, for faster training)

### Quick Setup

1. **Clone the repository**
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
   ```

4. **Quick start setup**
   ```bash
   python quick_start.py
   ```

## 🎯 Usage

### 🚀 Quick Start
```bash
# Launch the web application
python main.py --web

# Or use Streamlit directly
streamlit run interface/streamlit_app.py
```

### 🎨 Interactive Drawing
1. Open the web application
2. Draw a digit in the canvas
3. See real-time predictions from all models
4. Compare confidence scores and probabilities
5. Switch between individual models and ensemble

### 🧠 Training Models
```bash
# Train all models
python main.py --train

# Train specific model
python training/train_models.py
```

### 🧪 Testing & Evaluation
```bash
# Test models on sample data
python main.py --test

# Generate visualizations
python main.py --visualize

# Run feature demo
python demo.py
```

### 🎮 Standalone Drawing Canvas
```bash
# Open Pygame drawing interface
python main.py --draw
```

## 📊 Model Performance

| Model | Accuracy | Training Time | Inference Time | Use Case |
|-------|----------|---------------|----------------|----------|
| **MLP** | ~98.5% | Fast | Very Fast | Quick predictions |
| **LeNet-5** | ~99.2% | Medium | Fast | Balanced performance |
| **ResNet** | ~99.4% | Slow | Medium | High accuracy |
| **Ensemble** | ~99.6% | - | Medium | Best overall |

## 🔧 Configuration

Edit `config.py` to customize:

### Model Parameters
```python
MLP_CONFIG = {
    'hidden_layers': [512, 256, 128],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 20
}
```

### Training Settings
```python
TRAINING_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,
    'early_stopping_patience': 10
}
```

### Ensemble Weights
```python
ENSEMBLE_CONFIG = {
    'weights': {
        'mlp': 0.2,
        'lenet5': 0.3,
        'resnet': 0.5
    }
}
```

## 📈 Advanced Features

### Data Augmentation
- **Random rotation** (±15°)
- **Random shift** (±2 pixels)
- **Random zoom** (±10%)
- **Gaussian noise** addition
- **Brightness/contrast** adjustment

### Hyperparameter Tuning
- Learning rate optimization
- Batch size variation
- Dropout rate tuning
- Regularization strength

### Visualization Capabilities
- Training progress monitoring
- Model comparison plots
- Feature map analysis
- Error analysis gallery

## 🎨 Web Interface Features

### Main Dashboard
- **Drawing Canvas**: Interactive digit drawing
- **Real-time Predictions**: Instant model outputs
- **Confidence Scores**: Probability distributions
- **Model Selection**: Choose individual or ensemble

### Advanced Analytics
- **Training Curves**: Loss and accuracy plots
- **Confusion Matrices**: Model performance analysis
- **Feature Maps**: CNN layer visualizations
- **Error Analysis**: Misclassified digit gallery

## 🔍 API Reference

### Model Creation
```python
from models.mlp_model import create_mlp_model
from models.lenet5_model import create_lenet5_model
from models.resnet_model import create_resnet_model

mlp = create_mlp_model()
lenet5 = create_lenet5_model()
resnet = create_resnet_model()
```

### Data Loading
```python
from data.data_loader import MNISTDataLoader

data_loader = MNISTDataLoader(batch_size=64, augment_train=True)
train_loader, val_loader, test_loader = data_loader.get_dataloaders()
```

### Training
```python
from training.train_models import main as train_main
train_main()  # Train all models
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** if applicable
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Created by Yann LeCun and Corinna Cortes
- **PyTorch Community**: For the excellent deep learning framework
- **Streamlit Team**: For the amazing web app framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Sneakyboss221/Digit-Classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sneakyboss221/Digit-Classifier/discussions)
- **Email**: Sneakyboss221@gmail.com

## 🚀 Roadmap

- [ ] Add more model architectures (VGG, DenseNet)
- [ ] Implement transfer learning
- [ ] Add mobile app version
- [ ] Real-time video processing
- [ ] Multi-language support
- [ ] Cloud deployment guides

---

⭐ **Star this repository if you find it helpful!**
