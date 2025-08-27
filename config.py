"""
Configuration file for the Ultimate MNIST Digit Recognition Project
"""

import os

# Data Configuration
DATA_CONFIG = {
    'mnist_train_size': 60000,
    'mnist_test_size': 10000,
    'image_size': 28,
    'num_classes': 10,
    'train_val_split': 0.9,  # 90% train, 10% validation
    'batch_size': 64,
    'num_workers': 4
}

# Model Configurations
MLP_CONFIG = {
    'input_size': 784,  # 28*28 flattened
    'hidden_layers': [512, 256, 128],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 20,
    'optimizer': 'adam'
}

LENET5_CONFIG = {
    'conv1_filters': 6,
    'conv2_filters': 16,
    'conv3_filters': 120,
    'fc1_size': 84,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 30,
    'optimizer': 'adam'
}

RESNET_CONFIG = {
    'initial_channels': 32,
    'num_blocks': [2, 2],  # Number of residual blocks in each layer
    'block_channels': [32, 64],
    'fc_size': 128,
    'dropout_rate': 0.5,
    'learning_rate': 0.001,
    'epochs': 50,
    'optimizer': 'adam'
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 15,  # degrees
    'shift_range': 2,      # pixels
    'zoom_range': 0.1,     # 10%
    'noise_std': 0.01,     # Gaussian noise standard deviation
    'brightness_range': 0.1,
    'contrast_range': 0.1
}

# Training Configuration
import torch
TRAINING_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_best_model': True,
    'early_stopping_patience': 10,
    'learning_rate_scheduler': True,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'weight_decay': 1e-4
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'weights': {
        'mlp': 0.2,
        'lenet5': 0.3,
        'resnet': 0.5
    },
    'voting_method': 'weighted_average'  # 'weighted_average' or 'majority_voting'
}

# Interface Configuration
INTERFACE_CONFIG = {
    'canvas_size': 280,  # 10x MNIST size for better drawing
    'brush_size': 15,
    'prediction_threshold': 0.5,
    'max_predictions': 3,
    'update_frequency': 0.5  # seconds
}

# Paths
PATHS = {
    'saved_models': 'saved_models',
    'logs': 'logs',
    'results': 'results',
    'data': 'data'
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/training.log'
}
