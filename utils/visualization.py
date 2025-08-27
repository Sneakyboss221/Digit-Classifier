"""
Visualization utilities for the MNIST digit recognition project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import VISUALIZATION_CONFIG

class Visualizer:
    """Class for creating various visualizations"""
    
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_training_curves(self, train_losses, train_accuracies, val_losses, val_accuracies, 
                           model_name="Model", save_path=None):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title(f'{model_name} - Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [str(i) for i in range(10)]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return cm
    
    def plot_model_comparison(self, results_dict, save_path=None):
        """Compare multiple models' performance"""
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['best_val_accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=self.colors[:len(models)])
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_maps(self, model, image_tensor, layer_name, save_path=None):
        """Plot feature maps from a specific layer"""
        model.eval()
        
        # Register hook to get feature maps
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        # Register hook
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        with torch.no_grad():
            _ = model(image_tensor)
        
        if not feature_maps:
            print(f"Layer {layer_name} not found")
            return
        
        # Get feature maps
        fm = feature_maps[0][0]  # First batch, first image
        
        # Plot feature maps
        num_features = min(fm.shape[0], 16)  # Show max 16 feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_features):
            axes[i].imshow(fm[i].cpu().numpy(), cmap='viridis')
            axes[i].set_title(f'Feature {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_features, 16):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_misclassified_samples(self, test_loader, model, device, num_samples=16, save_path=None):
        """Plot misclassified samples"""
        model.eval()
        misclassified = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                # Find misclassified samples
                misclassified_mask = pred != target
                if misclassified_mask.any():
                    for i in range(data.size(0)):
                        if misclassified_mask[i] and len(misclassified) < num_samples:
                            misclassified.append({
                                'image': data[i].cpu(),
                                'true': target[i].item(),
                                'pred': pred[i].item(),
                                'confidence': torch.exp(output[i]).max().item()
                            })
                
                if len(misclassified) >= num_samples:
                    break
        
        if not misclassified:
            print("No misclassified samples found")
            return
        
        # Plot misclassified samples
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, sample in enumerate(misclassified[:num_samples]):
            axes[i].imshow(sample['image'].squeeze(), cmap='gray')
            axes[i].set_title(f'True: {sample["true"]}, Pred: {sample["pred"]}\nConf: {sample["confidence"]:.3f}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(misclassified), num_samples):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Samples')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_ensemble_predictions(self, individual_predictions, ensemble_prediction, 
                                ensemble_probabilities, save_path=None):
        """Plot ensemble prediction breakdown"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual model predictions
        models = list(individual_predictions.keys())
        predictions = [individual_predictions[model]['predicted_class'] for model in models]
        confidences = [individual_predictions[model]['probabilities'][individual_predictions[model]['predicted_class']] 
                      for model in models]
        
        bars1 = ax1.bar(models, predictions, color=self.colors[:len(models)])
        ax1.set_title('Individual Model Predictions')
        ax1.set_ylabel('Predicted Digit')
        ax1.set_ylim(0, 9)
        
        # Add confidence labels
        for bar, conf in zip(bars1, confidences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{conf:.3f}', ha='center', va='bottom')
        
        # Ensemble probabilities
        digits = range(10)
        bars2 = ax2.bar(digits, ensemble_probabilities, color='skyblue')
        ax2.set_title('Ensemble Probabilities')
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability')
        ax2.set_xticks(digits)
        
        # Highlight ensemble prediction
        bars2[ensemble_prediction].set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_plot(self, train_losses, train_accuracies, val_losses, val_accuracies, model_name="Model"):
        """Create interactive plot using Plotly"""
        epochs = list(range(1, len(train_losses) + 1))
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{model_name} - Loss', f'{model_name} - Accuracy'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, mode='lines', name='Training Loss',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, mode='lines', name='Validation Loss',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Accuracy curves
        fig.add_trace(
            go.Scatter(x=epochs, y=train_accuracies, mode='lines', name='Training Accuracy',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_accuracies, mode='lines', name='Validation Accuracy',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{model_name} Training Curves',
            showlegend=True,
            height=500
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        
        return fig

def plot_sample_digits(dataset, num_samples=16, save_path=None):
    """Plot sample digits from the dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(num_samples):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Digit: {label}')
        axes[i].axis('off')
    
    plt.suptitle('Sample MNIST Digits')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
