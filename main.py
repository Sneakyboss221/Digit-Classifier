"""
Main entry point for the Ultimate MNIST Digit Recognition Project
"""

import argparse
import sys
import os
import torch


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Ultimate MNIST Digit Recognition Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    # Train all models
  python main.py --web                      # Launch Streamlit web app
  python main.py --draw                     # Open drawing canvas
  python main.py --test                     # Test models on sample data
  python main.py --visualize                # Generate visualizations
        """,
    )

    parser.add_argument(
        "--train", action="store_true", help="Train all models (MLP, LeNet-5, ResNet)"
    )
    parser.add_argument(
        "--web", action="store_true", help="Launch Streamlit web application"
    )
    parser.add_argument(
        "--draw", action="store_true", help="Open interactive drawing canvas"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test models on sample data"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate training visualizations"
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "lenet5", "resnet", "ensemble"],
        help="Specify which model to use for testing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (overrides config)",
    )

    args = parser.parse_args()

    # Check if any action is specified
    if not any([args.train, args.web, args.draw, args.test, args.visualize]):
        parser.print_help()
        return

    # Check CUDA availability
    if torch.cuda.is_available():
        print("🚀 CUDA is available! Training will use GPU.")
    else:
        print("⚠️  CUDA not available. Training will use CPU.")

    # Execute requested actions
    if args.train:
        print("🎯 Starting model training...")
        from training.train_models import main as train_main

        train_main()

    if args.web:
        print("🌐 Launching Streamlit web application...")
        try:
            import streamlit

            os.system("streamlit run interface/streamlit_app.py")
        except ImportError:
            print(
                "❌ Streamlit not installed. Please install it with: pip install streamlit"
            )

    if args.draw:
        print("🎨 Opening drawing canvas...")
        try:
            from interface.drawing_canvas import test_drawing_canvas

            test_drawing_canvas()
        except ImportError as e:
            print(f"❌ Error opening drawing canvas: {e}")

    if args.test:
        print("🧪 Testing models...")
        test_models(args.model)

    if args.visualize:
        print("📊 Generating visualizations...")
        generate_visualizations()


def test_models(model_type=None):
    """Test models on sample data"""
    try:
        from data.data_loader import MNISTDataLoader
        from models.mlp_model import create_mlp_model, predict_mlp
        from models.lenet5_model import create_lenet5_model, predict_lenet5
        from models.resnet_model import create_resnet_model, predict_resnet
        from models.ensemble_model import create_ensemble_model
        from config import TRAINING_CONFIG, PATHS

        # Load test data
        print("📊 Loading test data...")
        data_loader = MNISTDataLoader(batch_size=100, augment_train=False)
        _, _, test_loader = data_loader.get_dataloaders()

        device = torch.device(TRAINING_CONFIG["device"])

        # Test specific model or all models
        models_to_test = []
        if model_type == "mlp" or model_type is None:
            models_to_test.append(("MLP", "mlp"))
        if model_type == "lenet5" or model_type is None:
            models_to_test.append(("LeNet-5", "lenet5"))
        if model_type == "resnet" or model_type is None:
            models_to_test.append(("ResNet", "resnet"))

        results = {}

        for model_name, model_key in models_to_test:
            try:
                # Load model
                if model_key == "mlp":
                    model = create_mlp_model()
                    checkpoint = torch.load(
                        f"{PATHS['saved_models']}/mlp_model.pth", map_location=device
                    )
                elif model_key == "lenet5":
                    model = create_lenet5_model()
                    checkpoint = torch.load(
                        f"{PATHS['saved_models']}/lenet5_model.pth", map_location=device
                    )
                elif model_key == "resnet":
                    model = create_resnet_model()
                    checkpoint = torch.load(
                        f"{PATHS['saved_models']}/resnet_model.pth", map_location=device
                    )

                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                model.eval()

                # Test model
                correct = 0
                total = 0

                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)

                        if model_key == "mlp":
                            data = data.view(data.size(0), -1)

                        output = model(data)
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)

                accuracy = 100.0 * correct / total
                results[model_name] = accuracy
                print(f"✅ {model_name} Test Accuracy: {accuracy:.2f}%")

            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                results[model_name] = 0

        # Test ensemble if all individual models are available
        if model_type is None or model_type == "ensemble":
            try:
                ensemble_models = {}
                for model_name, model_key in [
                    ("MLP", "mlp"),
                    ("LeNet-5", "lenet5"),
                    ("ResNet", "resnet"),
                ]:
                    if model_key in results and results[model_name] > 0:
                        if model_key == "mlp":
                            ensemble_models["mlp"] = create_mlp_model()
                        elif model_key == "lenet5":
                            ensemble_models["lenet5"] = create_lenet5_model()
                        elif model_key == "resnet":
                            ensemble_models["resnet"] = create_resnet_model()

                        checkpoint = torch.load(
                            f"{PATHS['saved_models']}/{model_key}_model.pth",
                            map_location=device,
                        )
                        ensemble_models[model_key].load_state_dict(
                            checkpoint["model_state_dict"]
                        )
                        ensemble_models[model_key].to(device)

                if len(ensemble_models) >= 2:
                    ensemble = create_ensemble_model(
                        ensemble_models.get("mlp"),
                        ensemble_models.get("lenet5"),
                        ensemble_models.get("resnet"),
                    )

                    # Test ensemble
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(device), target.to(device)

                            pred_class, _, _ = ensemble.predict(data, device)

                            if pred_class == target.item():
                                correct += 1
                            total += 1

                    ensemble_accuracy = 100.0 * correct / total
                    results["Ensemble"] = ensemble_accuracy
                    print(f"✅ Ensemble Test Accuracy: {ensemble_accuracy:.2f}%")

            except Exception as e:
                print(f"❌ Error testing Ensemble: {e}")

        # Print summary
        print("\n📊 Test Results Summary:")
        print("=" * 40)
        for model_name, accuracy in results.items():
            print(f"{model_name:12}: {accuracy:6.2f}%")

    except Exception as e:
        print(f"❌ Error during testing: {e}")


def generate_visualizations():
    """Generate training visualizations"""
    try:
        from utils.visualization import Visualizer
        from config import PATHS

        print("📈 Generating visualizations...")

        # This would typically load saved training results and create visualizations
        # For now, we'll just create a sample visualization
        visualizer = Visualizer()

        # Create a sample plot
        import numpy as np

        epochs = range(1, 21)
        train_loss = [2.0 - 0.1 * i + np.random.normal(0, 0.05) for i in epochs]
        val_loss = [2.1 - 0.08 * i + np.random.normal(0, 0.1) for i in epochs]
        train_acc = [50 + 2.5 * i + np.random.normal(0, 1) for i in epochs]
        val_acc = [48 + 2.3 * i + np.random.normal(0, 2) for i in epochs]

        visualizer.plot_training_curves(
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            "Sample Model",
            save_path=f"{PATHS['results']}/sample_training_curves.png",
        )

        print("✅ Visualizations generated successfully!")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
