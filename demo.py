#!/usr/bin/env python3
"""
Demo script for the Ultimate MNIST Digit Recognition Project
This script demonstrates the key features of the project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os


def create_sample_digit(digit, size=28):
    """Create a sample digit image for demonstration"""
    # Create a blank image
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to basic drawing if not available
    try:
        # Try to use a system font
        font_size = size // 2
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw the digit
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size - text_width) // 2
    y = (size - text_height) // 2

    draw.text((x, y), text, fill=255, font=font)

    return np.array(img)


def demo_data_loading():
    """Demonstrate data loading functionality"""
    print("📊 Demo: Data Loading")
    print("-" * 40)

    try:
        from data.data_loader import MNISTDataLoader

        # Load data
        data_loader = MNISTDataLoader(batch_size=16, augment_train=False)
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()

        # Show sample data
        for data, target in train_loader:
            print(f"Batch shape: {data.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Sample targets: {target[:8].tolist()}")

            # Display first 8 images
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.ravel()

            for i in range(8):
                axes[i].imshow(data[i].squeeze(), cmap="gray")
                axes[i].set_title(f"Digit: {target[i].item()}")
                axes[i].axis("off")

            plt.suptitle("Sample MNIST Digits")
            plt.tight_layout()
            plt.show()
            break

        print("✅ Data loading demo completed!")

    except Exception as e:
        print(f"❌ Data loading demo failed: {e}")


def demo_model_creation():
    """Demonstrate model creation"""
    print("\n🧠 Demo: Model Creation")
    print("-" * 40)

    try:
        from models.mlp_model import create_mlp_model
        from models.lenet5_model import create_lenet5_model
        from models.resnet_model import create_resnet_model

        # Create models
        mlp_model = create_mlp_model()
        lenet5_model = create_lenet5_model()
        resnet_model = create_resnet_model()

        print(f"MLP parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")
        print(
            f"LeNet-5 parameters: {sum(p.numel() for p in lenet5_model.parameters()):,}"
        )
        print(
            f"ResNet parameters: {sum(p.numel() for p in resnet_model.parameters()):,}"
        )

        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dummy input
        dummy_input = torch.randn(1, 1, 28, 28)

        # Test MLP
        mlp_input = dummy_input.view(1, -1)
        mlp_output = mlp_model(mlp_input)

        # Test LeNet-5
        lenet5_output = lenet5_model(dummy_input)

        # Test ResNet
        resnet_output = resnet_model(dummy_input)

        print(f"MLP output shape: {mlp_output.shape}")
        print(f"LeNet-5 output shape: {lenet5_output.shape}")
        print(f"ResNet output shape: {resnet_output.shape}")

        print("✅ Model creation demo completed!")

    except Exception as e:
        print(f"❌ Model creation demo failed: {e}")


def demo_preprocessing():
    """Demonstrate image preprocessing"""
    print("\n🖼️ Demo: Image Preprocessing")
    print("-" * 40)

    try:
        from data.data_loader import Preprocessor

        # Create sample digits
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i, digit in enumerate(digits):
            # Create sample digit
            sample_digit = create_sample_digit(digit, size=100)

            # Preprocess
            preprocessed = Preprocessor.preprocess_drawn_image(sample_digit)

            # Display original and preprocessed
            axes[i].imshow(sample_digit, cmap="gray")
            axes[i].set_title(f"Original {digit}")
            axes[i].axis("off")

            axes[i + 5].imshow(preprocessed.squeeze(), cmap="gray")
            axes[i + 5].set_title(f"Preprocessed {digit}")
            axes[i + 5].axis("off")

        plt.suptitle("Image Preprocessing Demo")
        plt.tight_layout()
        plt.show()

        print("✅ Preprocessing demo completed!")

    except Exception as e:
        print(f"❌ Preprocessing demo failed: {e}")


def demo_ensemble():
    """Demonstrate ensemble functionality"""
    print("\n🎯 Demo: Ensemble Model")
    print("-" * 40)

    try:
        from models.ensemble_model import create_ensemble_model
        from models.mlp_model import create_mlp_model
        from models.lenet5_model import create_lenet5_model
        from models.resnet_model import create_resnet_model

        # Create individual models
        mlp_model = create_mlp_model()
        lenet5_model = create_lenet5_model()
        resnet_model = create_resnet_model()

        # Create ensemble
        ensemble = create_ensemble_model(mlp_model, lenet5_model, resnet_model)

        print(f"Ensemble weights: {ensemble.weights}")
        print(f"Voting method: {ensemble.voting_method}")

        # Test ensemble prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 1, 28, 28)

        pred_class, ensemble_probs, individual_preds = ensemble.predict(
            dummy_input, device
        )

        print(f"Ensemble prediction: {pred_class}")
        print(f"Ensemble probabilities: {ensemble_probs}")
        print(f"Individual predictions: {individual_preds}")

        print("✅ Ensemble demo completed!")

    except Exception as e:
        print(f"❌ Ensemble demo failed: {e}")


def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n📈 Demo: Visualization")
    print("-" * 40)

    try:
        from utils.visualization import Visualizer

        visualizer = Visualizer()

        # Create sample training curves
        epochs = range(1, 21)
        train_loss = [2.0 - 0.1 * i + np.random.normal(0, 0.05) for i in epochs]
        val_loss = [2.1 - 0.08 * i + np.random.normal(0, 0.1) for i in epochs]
        train_acc = [50 + 2.5 * i + np.random.normal(0, 1) for i in epochs]
        val_acc = [48 + 2.3 * i + np.random.normal(0, 2) for i in epochs]

        # Plot training curves
        visualizer.plot_training_curves(
            train_loss, train_acc, val_loss, val_acc, "Demo Model"
        )

        # Create sample confusion matrix
        y_true = np.random.randint(0, 10, 1000)
        y_pred = y_true.copy()
        # Add some errors
        error_indices = np.random.choice(1000, 50, replace=False)
        y_pred[error_indices] = np.random.randint(0, 10, 50)

        visualizer.plot_confusion_matrix(y_true, y_pred)

        print("✅ Visualization demo completed!")

    except Exception as e:
        print(f"❌ Visualization demo failed: {e}")


def main():
    """Main demo function"""
    print("🎬 Ultimate MNIST Digit Recognition - Demo")
    print("=" * 60)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")

    # Run demos
    demos = [
        demo_data_loading,
        demo_model_creation,
        demo_preprocessing,
        demo_ensemble,
        demo_visualization,
    ]

    for demo in demos:
        try:
            demo()
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Run quick start: python quick_start.py")
    print("2. Train models: python main.py --train")
    print("3. Launch web app: python main.py --web")
    print("4. Test models: python main.py --test")


if __name__ == "__main__":
    main()
