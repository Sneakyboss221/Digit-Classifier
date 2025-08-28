#!/usr/bin/env python3
"""
Quick Start Script for Ultimate MNIST Digit Recognition Project
This script provides an easy way to get started with the project.
"""

import os
import sys
import subprocess
import torch


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")

    # Map package names to their import names
    package_mapping = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scikit-learn": "sklearn",
        "streamlit": "streamlit",
        "opencv-python": "cv2",
        "pillow": "PIL",
        "pandas": "pandas",
        "plotly": "plotly",
        "tqdm": "tqdm",
        "pygame": "pygame",
    }

    missing_packages = []

    for package, import_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies are installed!")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\n🚀 Checking CUDA availability...")

    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        return True
    else:
        print("⚠️  CUDA not available. Training will use CPU.")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")

    directories = ["saved_models", "logs", "results", "data"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")


def download_mnist():
    """Download MNIST dataset"""
    print("\n📊 Downloading MNIST dataset...")

    try:
        import torchvision
        from torchvision import datasets

        # This will automatically download MNIST
        datasets.MNIST(root="./data", train=True, download=True)
        datasets.MNIST(root="./data", train=False, download=True)

        print("✅ MNIST dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading MNIST: {e}")
        return False


def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n🧪 Running quick test...")

    try:
        # Test imports
        from data.data_loader import MNISTDataLoader
        from models.mlp_model import create_mlp_model
        from models.lenet5_model import create_lenet5_model
        from models.resnet_model import create_resnet_model

        # Test data loading
        data_loader = MNISTDataLoader(batch_size=32, augment_train=False)
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()

        # Test model creation
        mlp_model = create_mlp_model()
        lenet5_model = create_lenet5_model()
        resnet_model = create_resnet_model()

        print("✅ All imports and model creation successful!")

        # Test with a small batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Test MLP
            mlp_input = data.view(data.size(0), -1)
            mlp_output = mlp_model(mlp_input)

            # Test LeNet-5
            lenet5_output = lenet5_model(data)

            # Test ResNet
            resnet_output = resnet_model(data)

            print(f"✅ Forward pass successful!")
            print(f"   MLP output shape: {mlp_output.shape}")
            print(f"   LeNet-5 output shape: {lenet5_output.shape}")
            print(f"   ResNet output shape: {resnet_output.shape}")
            break

        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Train the models:")
    print("   python main.py --train")
    print("\n2. Launch the web interface:")
    print("   python main.py --web")
    print("\n3. Test the models:")
    print("   python main.py --test")
    print("\n4. Open drawing canvas:")
    print("   python main.py --draw")
    print("\n5. Generate visualizations:")
    print("   python main.py --visualize")
    print("\n📚 For more information, see README.md")
    print("=" * 60)


def main():
    """Main quick start function"""
    print("🚀 Ultimate MNIST Digit Recognition - Quick Start")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return

    # Check CUDA
    check_cuda()

    # Create directories
    create_directories()

    # Download MNIST
    if not download_mnist():
        print("\n❌ Failed to download MNIST dataset.")
        return

    # Run quick test
    if not run_quick_test():
        print("\n❌ Quick test failed. Please check your installation.")
        return

    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
