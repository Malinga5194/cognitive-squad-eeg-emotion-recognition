"""
Main Training Script
EEG-Based Cognitive State Classification
Cognitive Squad - MSc AI Project

This script trains and evaluates all models:
1. Baseline: SVM + Random Forest (on extracted features)
2. EEGNet (lightweight CNN)
3. LSTM (temporal model)
4. CNN-LSTM Hybrid (flagship model)

Usage:
    python train.py --dataset DEAP --label valence
    python train.py --dataset DREAMER --label arousal
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import DATASET_CONFIG
from src.data.dataset import EEGDataset, EEGFeatureDataset
from src.data.features import extract_all_features
from src.models.eegnet import EEGNet
from src.models.lstm_model import EEGLSTM
from src.models.cnn_lstm import CNNLSTM
from src.utils.trainer import train_model, evaluate, get_device
from src.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_model_comparison,
)


def load_dataset(dataset_name: str, label_name: str):
    """Load the specified dataset."""
    label_names = DATASET_CONFIG["label_names"]
    label_index = label_names.index(label_name)

    if dataset_name == "DEAP":
        from src.data.load_deap import load_all_subjects
        data, labels = load_all_subjects(binary_labels=True)
    elif dataset_name == "DREAMER":
        from src.data.load_dreamer import load_dreamer
        data, labels = load_dreamer(binary_labels=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nDataset: {dataset_name}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Target label: {label_name} (index {label_index})")
    print(f"Class distribution: {np.bincount(labels[:, label_index])}")

    return data, labels, label_index


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train and evaluate SVM and Random Forest baselines."""
    results = {}

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVM
    print("\n--- Training SVM ---")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_preds = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average="weighted")
    print(f"SVM Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}")
    print(classification_report(y_test, svm_preds, target_names=["Low", "High"]))
    results["SVM"] = {"accuracy": svm_acc, "f1": svm_f1, "preds": svm_preds}

    # Random Forest
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted")
    print(f"Random Forest Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    print(classification_report(y_test, rf_preds, target_names=["Low", "High"]))
    results["Random Forest"] = {"accuracy": rf_acc, "f1": rf_f1, "preds": rf_preds}

    return results


def train_deep_models(
    data_train, data_test, labels_train, labels_test,
    label_index, n_channels, n_samples, device,
):
    """Train and evaluate deep learning models."""
    results = {}

    # Create datasets
    train_dataset = EEGDataset(data_train, labels_train, label_index)
    test_dataset = EEGDataset(data_test, labels_test, label_index)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- EEGNet ---
    print("\n" + "=" * 60)
    print("Training EEGNet")
    eegnet = EEGNet(
        n_channels=n_channels, n_samples=n_samples,
        n_classes=2, dropout_rate=0.5,
    )
    eegnet_history = train_model(
        eegnet, train_loader, test_loader,
        n_epochs=50, learning_rate=1e-3,
        device=device, model_name="EEGNet",
    )
    _, eegnet_acc, eegnet_preds, eegnet_labels = evaluate(
        eegnet, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    eegnet_f1 = f1_score(eegnet_labels, eegnet_preds, average="weighted")
    results["EEGNet"] = {
        "accuracy": eegnet_acc, "f1": eegnet_f1,
        "preds": eegnet_preds, "labels": eegnet_labels,
        "history": eegnet_history,
    }

    # --- LSTM ---
    print("\n" + "=" * 60)
    print("Training LSTM")
    lstm = EEGLSTM(
        n_channels=n_channels, hidden_size=128,
        num_layers=2, n_classes=2, dropout=0.3,
    )
    lstm_history = train_model(
        lstm, train_loader, test_loader,
        n_epochs=50, learning_rate=1e-3,
        device=device, model_name="LSTM",
    )
    _, lstm_acc, lstm_preds, lstm_labels = evaluate(
        lstm, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    lstm_f1 = f1_score(lstm_labels, lstm_preds, average="weighted")
    results["LSTM"] = {
        "accuracy": lstm_acc, "f1": lstm_f1,
        "preds": lstm_preds, "labels": lstm_labels,
        "history": lstm_history,
    }

    # --- CNN-LSTM Hybrid ---
    print("\n" + "=" * 60)
    print("Training CNN-LSTM Hybrid")
    cnn_lstm = CNNLSTM(
        n_channels=n_channels, n_classes=2,
        cnn_filters=[32, 64], lstm_hidden=64,
        lstm_layers=1, dropout=0.3,
    )
    cnn_lstm_history = train_model(
        cnn_lstm, train_loader, test_loader,
        n_epochs=50, learning_rate=1e-3,
        device=device, model_name="CNN-LSTM",
    )
    _, cnn_lstm_acc, cnn_lstm_preds, cnn_lstm_labels = evaluate(
        cnn_lstm, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    cnn_lstm_f1 = f1_score(cnn_lstm_labels, cnn_lstm_preds, average="weighted")
    results["CNN-LSTM"] = {
        "accuracy": cnn_lstm_acc, "f1": cnn_lstm_f1,
        "preds": cnn_lstm_preds, "labels": cnn_lstm_labels,
        "history": cnn_lstm_history,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="EEG-Based Cognitive State Classification"
    )
    parser.add_argument(
        "--dataset", type=str, default="DEAP",
        choices=["DEAP", "DREAMER"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--label", type=str, default="valence",
        help="Target label (valence, arousal, dominance, liking)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs",
    )
    args = parser.parse_args()

    # Load data
    data, labels, label_index = load_dataset(args.dataset, args.label)

    n_channels = data.shape[1]
    n_samples = data.shape[2]

    # Split data: 80% train, 20% test
    (data_train, data_test,
     labels_train, labels_test) = train_test_split(
        data, labels, test_size=0.2, random_state=42,
        stratify=labels[:, label_index],
    )

    print(f"\nTrain set: {data_train.shape[0]} trials")
    print(f"Test set:  {data_test.shape[0]} trials")

    # --- Baseline Models (feature-based) ---
    print("\n" + "=" * 60)
    print("PHASE 1: Feature Extraction + Baseline Models")
    print("=" * 60)

    features_train = extract_all_features(
        data_train, DATASET_CONFIG["sampling_rate"]
    )
    features_test = extract_all_features(
        data_test, DATASET_CONFIG["sampling_rate"]
    )

    y_train = labels_train[:, label_index]
    y_test = labels_test[:, label_index]

    baseline_results = train_baseline_models(
        features_train, features_test, y_train, y_test
    )

    # --- Deep Learning Models ---
    print("\n" + "=" * 60)
    print("PHASE 2: Deep Learning Models")
    print("=" * 60)

    device = get_device()
    deep_results = train_deep_models(
        data_train, data_test, labels_train, labels_test,
        label_index, n_channels, n_samples, device,
    )

    # --- Results Summary ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    all_results = {**baseline_results, **deep_results}
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 42)
    for name, res in all_results.items():
        print(f"{name:<20} {res['accuracy']:>10.4f} {res['f1']:>10.4f}")

    # --- Generate Visualizations ---
    print("\nGenerating visualizations...")

    # Training curves for deep models
    for name in ["EEGNet", "LSTM", "CNN-LSTM"]:
        if name in deep_results and "history" in deep_results[name]:
            plot_training_history(deep_results[name]["history"], name)

    # Confusion matrices
    for name, res in all_results.items():
        if "preds" in res:
            true_labels = res.get("labels", y_test)
            plot_confusion_matrix(true_labels, res["preds"], model_name=name)

    # Model comparison
    comparison = {
        name: {"accuracy": res["accuracy"], "f1": res["f1"]}
        for name, res in all_results.items()
    }
    plot_model_comparison(comparison, metric="accuracy")
    plot_model_comparison(comparison, metric="f1")

    print("\nAll done! Results saved to results/figures/")


if __name__ == "__main__":
    main()
