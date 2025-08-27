"""
Ensemble model that combines predictions from MLP, LeNet-5, and ResNet models
"""

import torch
import numpy as np
from config import ENSEMBLE_CONFIG

class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models
    """
    
    def __init__(self, models, weights=None, voting_method='weighted_average'):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of models {'mlp': model, 'lenet5': model, 'resnet': model}
            weights: Dictionary of weights for each model
            voting_method: 'weighted_average' or 'majority_voting'
        """
        self.models = models
        self.weights = weights or ENSEMBLE_CONFIG['weights']
        self.voting_method = voting_method or ENSEMBLE_CONFIG['voting_method']
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def predict(self, image_tensor, device):
        """
        Make ensemble prediction
        
        Args:
            image_tensor: Input image tensor
            device: Device to run inference on
        
        Returns:
            predicted_class: Predicted digit class
            ensemble_probabilities: Combined probabilities
            individual_predictions: Dictionary of individual model predictions
        """
        individual_predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name == 'mlp':
                # Flatten for MLP
                if image_tensor.dim() > 2:
                    mlp_input = image_tensor.view(image_tensor.size(0), -1)
                else:
                    mlp_input = image_tensor
                pred_class, probs = self._predict_mlp(model, mlp_input, device)
            elif model_name == 'lenet5':
                pred_class, probs = self._predict_lenet5(model, image_tensor, device)
            elif model_name == 'resnet':
                pred_class, probs = self._predict_resnet(model, image_tensor, device)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            individual_predictions[model_name] = {
                'predicted_class': pred_class,
                'probabilities': probs
            }
        
        # Combine predictions
        if self.voting_method == 'weighted_average':
            predicted_class, ensemble_probabilities = self._weighted_average(individual_predictions)
        elif self.voting_method == 'majority_voting':
            predicted_class, ensemble_probabilities = self._majority_voting(individual_predictions)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        return predicted_class, ensemble_probabilities, individual_predictions
    
    def _predict_mlp(self, model, image_tensor, device):
        """Make prediction using MLP model"""
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = torch.exp(output)
            predicted_class = output.argmax(dim=1)
            return predicted_class.item(), probabilities[0].cpu().numpy()
    
    def _predict_lenet5(self, model, image_tensor, device):
        """Make prediction using LeNet-5 model"""
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = torch.exp(output)
            predicted_class = output.argmax(dim=1)
            return predicted_class.item(), probabilities[0].cpu().numpy()
    
    def _predict_resnet(self, model, image_tensor, device):
        """Make prediction using ResNet model"""
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = torch.exp(output)
            predicted_class = output.argmax(dim=1)
            return predicted_class.item(), probabilities[0].cpu().numpy()
    
    def _weighted_average(self, individual_predictions):
        """Combine predictions using weighted average"""
        ensemble_probabilities = np.zeros(10)
        
        for model_name, prediction in individual_predictions.items():
            weight = self.weights[model_name]
            ensemble_probabilities += weight * prediction['probabilities']
        
        predicted_class = np.argmax(ensemble_probabilities)
        
        return predicted_class, ensemble_probabilities
    
    def _majority_voting(self, individual_predictions):
        """Combine predictions using majority voting"""
        # Count votes for each class
        vote_counts = np.zeros(10)
        
        for model_name, prediction in individual_predictions.items():
            predicted_class = prediction['predicted_class']
            vote_counts[predicted_class] += 1
        
        # Get majority class
        predicted_class = np.argmax(vote_counts)
        
        # Create probability distribution (normalize vote counts)
        ensemble_probabilities = vote_counts / len(individual_predictions)
        
        return predicted_class, ensemble_probabilities
    
    def get_model_confidence(self, individual_predictions):
        """Calculate confidence scores for each model"""
        confidences = {}
        
        for model_name, prediction in individual_predictions.items():
            probabilities = prediction['probabilities']
            predicted_class = prediction['predicted_class']
            confidence = probabilities[predicted_class]
            confidences[model_name] = confidence
        
        return confidences
    
    def get_ensemble_confidence(self, ensemble_probabilities):
        """Calculate ensemble confidence"""
        predicted_class = np.argmax(ensemble_probabilities)
        return ensemble_probabilities[predicted_class]

class EnsembleTrainer:
    """Trainer for ensemble model (trains individual models)"""
    
    def __init__(self, models, device, configs=None):
        self.models = models
        self.device = device
        self.configs = configs or {}
        
        # Import trainers
        from models.mlp_model import MLPTrainer
        from models.lenet5_model import LeNet5Trainer
        from models.resnet_model import ResNetTrainer
        
        # Create trainers
        self.trainers = {}
        if 'mlp' in models:
            self.trainers['mlp'] = MLPTrainer(
                models['mlp'], device, self.configs.get('mlp')
            )
        
        if 'lenet5' in models:
            self.trainers['lenet5'] = LeNet5Trainer(
                models['lenet5'], device, self.configs.get('lenet5')
            )
        
        if 'resnet' in models:
            self.trainers['resnet'] = ResNetTrainer(
                models['resnet'], device, self.configs.get('resnet')
            )
    
    def train_all_models(self, train_loader, val_loader, epochs=None):
        """Train all models in the ensemble"""
        results = {}
        
        for model_name, trainer in self.trainers.items():
            print(f"\nTraining {model_name.upper()} model...")
            
            model_epochs = epochs or trainer.config['epochs']
            model_results = self._train_model(
                trainer, train_loader, val_loader, model_epochs
            )
            
            results[model_name] = model_results
            
            # Save model
            trainer.save_model(f"saved_models/{model_name}_model.pth")
        
        return results
    
    def _train_model(self, trainer, train_loader, val_loader, epochs):
        """Train a single model"""
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = trainer.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = trainer.validate(val_loader)
            
            # Learning rate scheduling
            if hasattr(trainer, 'scheduler'):
                trainer.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def evaluate_ensemble(self, test_loader):
        """Evaluate ensemble performance on test set"""
        ensemble_model = EnsembleModel(self.models)
        
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Get ensemble prediction
            pred_class, ensemble_probs, individual_preds = ensemble_model.predict(data, self.device)
            
            # Check if correct
            if pred_class == target.item():
                correct += 1
            total += 1
            
            all_predictions.append(pred_class)
            all_targets.append(target.item())
        
        accuracy = 100. * correct / total
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets
        }

def create_ensemble_model(mlp_model, lenet5_model, resnet_model, weights=None):
    """Factory function to create ensemble model"""
    models = {}
    if mlp_model:
        models['mlp'] = mlp_model
    if lenet5_model:
        models['lenet5'] = lenet5_model
    if resnet_model:
        models['resnet'] = resnet_model
    
    return EnsembleModel(models, weights)
