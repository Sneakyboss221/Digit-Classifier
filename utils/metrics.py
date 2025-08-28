"""
Evaluation metrics for the MNIST digit recognition project
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import torch


class MetricsCalculator:
    """Class for calculating various evaluation metrics"""

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calculate accuracy score"""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision(y_true, y_pred, average="macro"):
        """Calculate precision score"""
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def calculate_recall(y_true, y_pred, average="macro"):
        """Calculate recall score"""
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def calculate_f1(y_true, y_pred, average="macro"):
        """Calculate F1 score"""
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred):
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(y_true, y_pred, target_names=None):
        """Get detailed classification report"""
        if target_names is None:
            target_names = [str(i) for i in range(10)]
        return classification_report(y_true, y_pred, target_names=target_names)

    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        """Calculate all metrics at once"""
        return {
            "accuracy": MetricsCalculator.calculate_accuracy(y_true, y_pred),
            "precision_macro": MetricsCalculator.calculate_precision(
                y_true, y_pred, "macro"
            ),
            "precision_micro": MetricsCalculator.calculate_precision(
                y_true, y_pred, "micro"
            ),
            "precision_weighted": MetricsCalculator.calculate_precision(
                y_true, y_pred, "weighted"
            ),
            "recall_macro": MetricsCalculator.calculate_recall(y_true, y_pred, "macro"),
            "recall_micro": MetricsCalculator.calculate_recall(y_true, y_pred, "micro"),
            "recall_weighted": MetricsCalculator.calculate_recall(
                y_true, y_pred, "weighted"
            ),
            "f1_macro": MetricsCalculator.calculate_f1(y_true, y_pred, "macro"),
            "f1_micro": MetricsCalculator.calculate_f1(y_true, y_pred, "micro"),
            "f1_weighted": MetricsCalculator.calculate_f1(y_true, y_pred, "weighted"),
            "confusion_matrix": MetricsCalculator.calculate_confusion_matrix(
                y_true, y_pred
            ),
        }


def evaluate_model_predictions(model, test_loader, device, model_name="Model"):
    """Evaluate model predictions and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Handle different model input requirements
            if hasattr(model, "fc1"):  # MLP model
                data = data.view(data.size(0), -1)

            output = model(data)
            probabilities = torch.exp(output)
            pred = output.argmax(dim=1)

            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate metrics
    metrics = MetricsCalculator.calculate_all_metrics(all_targets, all_predictions)

    # Add additional metrics
    metrics["model_name"] = model_name
    metrics["total_samples"] = len(all_targets)
    metrics["predictions"] = all_predictions
    metrics["targets"] = all_targets
    metrics["probabilities"] = all_probabilities

    return metrics


def print_metrics_summary(metrics):
    """Print a formatted summary of metrics"""
    print(f"\n{'='*50}")
    print(f"📊 {metrics['model_name']} Evaluation Results")
    print(f"{'='*50}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Micro): {metrics['precision_micro']:.4f}")
    print(f"Recall (Micro): {metrics['recall_micro']:.4f}")
    print(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"{'='*50}")


def compare_models_metrics(metrics_list):
    """Compare metrics across multiple models"""
    print(f"\n{'='*80}")
    print("📈 MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print(f"{'-'*80}")

    for metrics in metrics_list:
        print(
            f"{metrics['model_name']:<15} "
            f"{metrics['accuracy']:<10.4f} "
            f"{metrics['precision_macro']:<10.4f} "
            f"{metrics['recall_macro']:<10.4f} "
            f"{metrics['f1_macro']:<10.4f}"
        )

    print(f"{'='*80}")

    # Find best model for each metric
    best_accuracy = max(metrics_list, key=lambda x: x["accuracy"])
    best_precision = max(metrics_list, key=lambda x: x["precision_macro"])
    best_recall = max(metrics_list, key=lambda x: x["recall_macro"])
    best_f1 = max(metrics_list, key=lambda x: x["f1_macro"])

    print(f"\n🏆 Best Models:")
    print(
        f"   Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})"
    )
    print(
        f"   Precision: {best_precision['model_name']} ({best_precision['precision_macro']:.4f})"
    )
    print(f"   Recall: {best_recall['model_name']} ({best_recall['recall_macro']:.4f})")
    print(f"   F1 Score: {best_f1['model_name']} ({best_f1['f1_macro']:.4f})")


def calculate_confidence_metrics(probabilities, targets):
    """Calculate confidence-related metrics"""
    max_probs = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    correct_predictions = predicted_classes == targets

    # Average confidence for correct vs incorrect predictions
    correct_confidence = (
        np.mean(max_probs[correct_predictions]) if np.any(correct_predictions) else 0
    )
    incorrect_confidence = (
        np.mean(max_probs[~correct_predictions]) if np.any(~correct_predictions) else 0
    )

    # Confidence calibration metrics
    confidence_bins = np.linspace(0, 1, 11)
    calibration_errors = []

    for i in range(len(confidence_bins) - 1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
        if np.any(mask):
            bin_confidence = np.mean(max_probs[mask])
            bin_accuracy = np.mean(correct_predictions[mask])
            calibration_errors.append(abs(bin_confidence - bin_accuracy))

    avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0

    return {
        "correct_confidence": correct_confidence,
        "incorrect_confidence": incorrect_confidence,
        "confidence_gap": correct_confidence - incorrect_confidence,
        "avg_calibration_error": avg_calibration_error,
        "overall_confidence": np.mean(max_probs),
    }
