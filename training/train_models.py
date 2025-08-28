"""
Main training script for all MNIST models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import MNISTDataLoader
from models.mlp_model import create_mlp_model, MLPTrainer
from models.lenet5_model import create_lenet5_model, LeNet5Trainer
from models.resnet_model import create_resnet_model, ResNetTrainer
from models.ensemble_model import EnsembleTrainer
from utils.visualization import Visualizer
from config import TRAINING_CONFIG, PATHS, MLP_CONFIG, LENET5_CONFIG, RESNET_CONFIG


def train_single_model(
    model_name, model, trainer, train_loader, val_loader, epochs, device
):
    """Train a single model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} model...")
    print(f"{'='*50}")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader)

        # Validation
        val_loss, val_acc = trainer.validate(val_loader)

        # Learning rate scheduling
        if hasattr(trainer, "scheduler"):
            trainer.scheduler.step(val_loss)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Early stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            # Save best model
            trainer.save_model(
                f"{PATHS['saved_models']}/{model_name.lower()}_model.pth"
            )
        else:
            patience_counter += 1

        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        # Early stopping check
        if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_accuracy,
    }


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate a trained model on test set"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            data, target = data.to(device), target.to(device)

            # Handle different model input requirements
            if model_name == "MLP":
                data = data.view(data.size(0), -1)

            output = model(data)
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100.0 * correct / total
    print(f"{model_name} Test Accuracy: {accuracy:.2f}%")

    return accuracy, all_predictions, all_targets


def main():
    """Main training function"""
    print("🚀 Starting MNIST Model Training")
    print(f"Device: {TRAINING_CONFIG['device']}")

    # Create directories
    os.makedirs(PATHS["saved_models"], exist_ok=True)
    os.makedirs(PATHS["results"], exist_ok=True)

    # Load data
    print("\n📊 Loading MNIST dataset...")
    data_loader = MNISTDataLoader(batch_size=64, augment_train=True)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()

    dataset_info = data_loader.get_dataset_info()
    print(f"Dataset info: {dataset_info}")

    # Initialize device
    device = torch.device(TRAINING_CONFIG["device"])

    # Initialize visualizer
    visualizer = Visualizer()

    # Training results storage
    all_results = {}

    # Train MLP
    print("\n🧠 Training MLP model...")
    mlp_model = create_mlp_model()
    mlp_trainer = MLPTrainer(mlp_model, device, MLP_CONFIG)

    mlp_results = train_single_model(
        "MLP",
        mlp_model,
        mlp_trainer,
        train_loader,
        val_loader,
        MLP_CONFIG["epochs"],
        device,
    )
    all_results["MLP"] = mlp_results

    # Evaluate MLP
    mlp_accuracy, mlp_preds, mlp_targets = evaluate_model(
        mlp_model, test_loader, device, "MLP"
    )

    # Train LeNet-5
    print("\n🔍 Training LeNet-5 model...")
    lenet5_model = create_lenet5_model()
    lenet5_trainer = LeNet5Trainer(lenet5_model, device, LENET5_CONFIG)

    lenet5_results = train_single_model(
        "LeNet-5",
        lenet5_model,
        lenet5_trainer,
        train_loader,
        val_loader,
        LENET5_CONFIG["epochs"],
        device,
    )
    all_results["LeNet-5"] = lenet5_results

    # Evaluate LeNet-5
    lenet5_accuracy, lenet5_preds, lenet5_targets = evaluate_model(
        lenet5_model, test_loader, device, "LeNet-5"
    )

    # Train ResNet
    print("\n🏗️ Training ResNet model...")
    resnet_model = create_resnet_model()
    resnet_trainer = ResNetTrainer(resnet_model, device, RESNET_CONFIG)

    resnet_results = train_single_model(
        "ResNet",
        resnet_model,
        resnet_trainer,
        train_loader,
        val_loader,
        RESNET_CONFIG["epochs"],
        device,
    )
    all_results["ResNet"] = resnet_results

    # Evaluate ResNet
    resnet_accuracy, resnet_preds, resnet_targets = evaluate_model(
        resnet_model, test_loader, device, "ResNet"
    )

    # Create and evaluate ensemble
    print("\n🎯 Creating ensemble model...")
    ensemble_models = {"mlp": mlp_model, "lenet5": lenet5_model, "resnet": resnet_model}

    ensemble_trainer = EnsembleTrainer(ensemble_models, device)
    ensemble_results = ensemble_trainer.evaluate_ensemble(test_loader)

    print(f"Ensemble Test Accuracy: {ensemble_results['accuracy']:.2f}%")

    # Visualizations
    print("\n📈 Creating visualizations...")

    # Training curves
    for model_name, results in all_results.items():
        visualizer.plot_training_curves(
            results["train_losses"],
            results["train_accuracies"],
            results["val_losses"],
            results["val_accuracies"],
            model_name,
            save_path=f"{PATHS['results']}/{model_name.lower()}_training_curves.png",
        )

    # Model comparison
    comparison_data = {
        "MLP": {"best_val_accuracy": mlp_results["best_val_accuracy"]},
        "LeNet-5": {"best_val_accuracy": lenet5_results["best_val_accuracy"]},
        "ResNet": {"best_val_accuracy": resnet_results["best_val_accuracy"]},
    }

    visualizer.plot_model_comparison(
        comparison_data, save_path=f"{PATHS['results']}/model_comparison.png"
    )

    # Confusion matrices
    visualizer.plot_confusion_matrix(
        mlp_targets, mlp_preds, save_path=f"{PATHS['results']}/mlp_confusion_matrix.png"
    )

    visualizer.plot_confusion_matrix(
        lenet5_targets,
        lenet5_preds,
        save_path=f"{PATHS['results']}/lenet5_confusion_matrix.png",
    )

    visualizer.plot_confusion_matrix(
        resnet_targets,
        resnet_preds,
        save_path=f"{PATHS['results']}/resnet_confusion_matrix.png",
    )

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"MLP Test Accuracy: {mlp_accuracy:.2f}%")
    print(f"LeNet-5 Test Accuracy: {lenet5_accuracy:.2f}%")
    print(f"ResNet Test Accuracy: {resnet_accuracy:.2f}%")
    print(f"Ensemble Test Accuracy: {ensemble_results['accuracy']:.2f}%")
    print("\n📁 Results saved to:")
    print(f"   Models: {PATHS['saved_models']}")
    print(f"   Visualizations: {PATHS['results']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
