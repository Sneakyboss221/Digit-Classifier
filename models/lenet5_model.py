"""
LeNet-5 CNN model for MNIST digit recognition
Based on the implementation from Jalalbaim/MNIST-Digit-recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LENET5_CONFIG

class LeNet5Model(nn.Module):
    """
    LeNet-5 Convolutional Neural Network for MNIST digit recognition
    
    Architecture:
    - Conv1: 1x6x5x5 -> MaxPool2D -> Conv2: 6x16x5x5 -> MaxPool2D
    - Conv3: 16x120x1x1 -> FC1: 120x84 -> FC2: 84x10
    """
    
    def __init__(self, config=None):
        super(LeNet5Model, self).__init__()
        self.config = config or LENET5_CONFIG
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, self.config['conv1_filters'], kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.config['conv1_filters'], self.config['conv2_filters'], 
                              kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.config['conv2_filters'], self.config['conv3_filters'], 
                              kernel_size=5, stride=1, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.config['conv3_filters'], self.config['fc1_size'])
        self.fc2 = nn.Linear(self.config['fc1_size'], 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(self.config['conv1_filters'])
        self.bn2 = nn.BatchNorm2d(self.config['conv2_filters'])
        self.bn3 = nn.BatchNorm2d(self.config['conv3_filters'])
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_features(self, x):
        """Extract features from the last convolutional layer"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        return x.view(x.size(0), -1)

class LeNet5Trainer:
    """Trainer class for LeNet-5 model"""
    
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or LENET5_CONFIG
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=1e-4
            )
        elif self.config['optimizer'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('config', self.config)

def create_lenet5_model(config=None):
    """Factory function to create LeNet-5 model"""
    return LeNet5Model(config)

def predict_lenet5(model, image_tensor, device):
    """Make prediction using LeNet-5 model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        output = model(image_tensor)
        probabilities = torch.exp(output)
        predicted_class = output.argmax(dim=1)
        
        return predicted_class.item(), probabilities[0].cpu().numpy()
