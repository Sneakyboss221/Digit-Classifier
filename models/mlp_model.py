"""
MLP (Multi-Layer Perceptron) model for MNIST digit recognition
Based on the implementation from aakashjhawar/handwritten-digit-recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MLP_CONFIG

class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron for MNIST digit recognition
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden layers: 512 -> 256 -> 128
    - Output: 10 (digit classes)
    - Dropout for regularization
    """
    
    def __init__(self, config=None):
        super(MLPModel, self).__init__()
        self.config = config or MLP_CONFIG
        
        # Define layers
        self.fc1 = nn.Linear(self.config['input_size'], self.config['hidden_layers'][0])
        self.fc2 = nn.Linear(self.config['hidden_layers'][0], self.config['hidden_layers'][1])
        self.fc3 = nn.Linear(self.config['hidden_layers'][1], self.config['hidden_layers'][2])
        self.fc4 = nn.Linear(self.config['hidden_layers'][2], 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        
        # Batch normalization for better training
        self.bn1 = nn.BatchNorm1d(self.config['hidden_layers'][0])
        self.bn2 = nn.BatchNorm1d(self.config['hidden_layers'][1])
        self.bn3 = nn.BatchNorm1d(self.config['hidden_layers'][2])
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_features(self, x):
        """Extract features from the last hidden layer"""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        
        return x

class MLPTrainer:
    """Trainer class for MLP model"""
    
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or MLP_CONFIG
        
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
            
            # Flatten data for MLP
            data = data.view(data.size(0), -1)
            
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
                
                # Flatten data for MLP
                data = data.view(data.size(0), -1)
                
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

def create_mlp_model(config=None):
    """Factory function to create MLP model"""
    return MLPModel(config)

def predict_mlp(model, image_tensor, device):
    """Make prediction using MLP model"""
    model.eval()
    with torch.no_grad():
        # Flatten and move to device
        if image_tensor.dim() > 2:
            image_tensor = image_tensor.view(image_tensor.size(0), -1)
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        output = model(image_tensor)
        probabilities = torch.exp(output)
        predicted_class = output.argmax(dim=1)
        
        return predicted_class.item(), probabilities[0].cpu().numpy()
