"""
ResNet model for MNIST digit recognition
Based on the implementation from xEC40/MNIST-ResNet-digit-recognizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import RESNET_CONFIG

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add shortcut
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNetModel(nn.Module):
    """
    ResNet model for MNIST digit recognition
    
    Architecture:
    - Initial conv: 1x32x3x3
    - Residual layers with increasing channels
    - Global average pooling
    - Fully connected layers
    """
    
    def __init__(self, config=None):
        super(ResNetModel, self).__init__()
        self.config = config or RESNET_CONFIG
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, self.config['initial_channels'], 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.config['initial_channels'])
        
        # Residual layers
        self.layer1 = self._make_layer(
            self.config['initial_channels'], 
            self.config['block_channels'][0], 
            self.config['num_blocks'][0], 
            stride=1
        )
        
        self.layer2 = self._make_layer(
            self.config['block_channels'][0], 
            self.config['block_channels'][1], 
            self.config['num_blocks'][1], 
            stride=1  # Changed from 2 to 1 to avoid making feature map too small
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.config['block_channels'][1], self.config['fc_size'])
        self.fc2 = nn.Linear(self.config['fc_size'], 10)
        
        # Dropout
        self.dropout = nn.Dropout(self.config['dropout_rate'])
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer of residual blocks"""
        layers = []
        
        # First block with specified stride
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks with stride 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_features(self, x):
        """Extract features from the last residual layer"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

class ResNetTrainer:
    """Trainer class for ResNet model"""
    
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or RESNET_CONFIG
        
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

def create_resnet_model(config=None):
    """Factory function to create ResNet model"""
    return ResNetModel(config)

def predict_resnet(model, image_tensor, device):
    """Make prediction using ResNet model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        output = model(image_tensor)
        probabilities = torch.exp(output)
        predicted_class = output.argmax(dim=1)
        
        return predicted_class.item(), probabilities[0].cpu().numpy()
