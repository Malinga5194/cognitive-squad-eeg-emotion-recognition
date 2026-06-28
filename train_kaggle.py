# Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
# MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).
# Developed for the Artificial Intelligence module group project.
# All rights reserved.

"""
Training Script for Kaggle EEG Brainwave Emotion Dataset
EEG-Based Cognitive State Classification
Cognitive Squad - MSc AI Project

Models trained:
1. SVM (baseline)
2. Random Forest (baseline)
3. MLP Neural Network (PyTorch)
4. LSTM on feature sequences (PyTorch)
5. CNN-1D (PyTorch)
6. Transformer (PyTorch) - State-of-the-art attention-based model

Usage:
    python train_kaggle.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from src.data.load_kaggle_eeg import load_kaggle_eeg
from src.utils.trainer import get_device
from src.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_model_comparison,
)


# ============================================================
# MLP Model for feature-based classification
# ============================================================
class EmotionMLP(nn.Module):
    """Multi-Layer Perceptron for EEG emotion classification."""

    def __init__(self, n_features: int, n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# LSTM Model - treats features as a sequence of channel groups
# ============================================================
class EmotionLSTM(nn.Module):
    """LSTM that treats feature groups as a temporal sequence."""

    def __init__(
        self, n_features: int, n_classes: int = 3,
        hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3,
    ):
        super().__init__()
        # Reshape features into (seq_len, input_size) for LSTM
        # We'll split 2548 features into chunks of ~50
        self.chunk_size = 50
        self.seq_len = n_features // self.chunk_size
        self.remainder = n_features % self.chunk_size

        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # Truncate to fit evenly into chunks
        usable = self.seq_len * self.chunk_size
        x = x[:, :usable]
        # Reshape: (batch, seq_len, chunk_size)
        x = x.view(x.size(0), self.seq_len, self.chunk_size)
        lstm_out, (h_n, _) = self.lstm(x)
        # Concatenate last hidden states from both directions
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.dropout(hidden)
        return self.fc(out)


# ============================================================
# 1D CNN Model
# ============================================================
class EmotionCNN1D(nn.Module):
    """1D CNN that treats features as a 1D signal."""

    def __init__(self, n_features: int, n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ============================================================
# Training utilities
# ============================================================
def train_pytorch_model(
    model, X_train, y_train, X_test, y_test,
    model_name, device, n_epochs=100, batch_size=32, lr=1e-3,
):
    """Train a PyTorch model and return results."""
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            train_correct += (out.argmax(1) == by).sum().item()
            train_total += bx.size(0)

        # Evaluate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
                val_correct += (out.argmax(1) == by).sum().item()
                val_total += bx.size(0)

        t_loss = train_loss / train_total
        t_acc = train_correct / train_total
        v_loss = val_loss / val_total
        v_acc = val_correct / val_total

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        scheduler.step(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train: {t_acc:.4f} | Val: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"results/models/{model_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best and get final predictions
    model.load_state_dict(
        torch.load(f"results/models/{model_name}_best.pth", weights_only=True)
    )
    model.eval()
    all_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            preds = model(bx).argmax(1).cpu().numpy()
            all_preds.extend(preds)

    preds = np.array(all_preds)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"\n{model_name} Best Test Accuracy: {acc:.4f}, F1: {f1:.4f}")

    return {
        "accuracy": acc, "f1": f1,
        "preds": preds, "history": history,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("EEG-Based Cognitive State Classification")
    print("Cognitive Squad - MSc AI Project")
    print("Dataset: Kaggle EEG Brainwave Feeling Emotions")
    print("=" * 60)

    # Load data
    data = load_kaggle_eeg(test_size=0.2, random_state=42, scale=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    class_names = data["class_names"]
    n_features = data["n_features"]
    n_classes = data["n_classes"]

    all_results = {}

    # ========================================
    # PHASE 1: Traditional ML Baselines
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Traditional Machine Learning Baselines")
    print("=" * 60)

    # SVM
    print("\n--- SVM (RBF Kernel) ---")
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average="weighted")
    print(f"Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}")
    print(classification_report(y_test, svm_preds, target_names=class_names))
    all_results["SVM"] = {"accuracy": svm_acc, "f1": svm_f1, "preds": svm_preds}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=30, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted")
    print(f"Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    print(classification_report(y_test, rf_preds, target_names=class_names))
    all_results["Random Forest"] = {"accuracy": rf_acc, "f1": rf_f1, "preds": rf_preds}

    # KNN
    print("\n--- K-Nearest Neighbors ---")
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)
    knn_f1 = f1_score(y_test, knn_preds, average="weighted")
    print(f"Accuracy: {knn_acc:.4f}, F1: {knn_f1:.4f}")
    print(classification_report(y_test, knn_preds, target_names=class_names))
    all_results["KNN"] = {"accuracy": knn_acc, "f1": knn_f1, "preds": knn_preds}

    # ========================================
    # PHASE 2: Deep Learning Models
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Deep Learning Models")
    print("=" * 60)

    device = get_device()

    # MLP
    mlp = EmotionMLP(n_features=n_features, n_classes=n_classes)
    mlp_results = train_pytorch_model(
        mlp, X_train, y_train, X_test, y_test,
        "MLP", device, n_epochs=100, lr=1e-3,
    )
    all_results["MLP"] = mlp_results

    # 1D CNN
    cnn = EmotionCNN1D(n_features=n_features, n_classes=n_classes)
    cnn_results = train_pytorch_model(
        cnn, X_train, y_train, X_test, y_test,
        "CNN-1D", device, n_epochs=100, lr=1e-3,
    )
    all_results["CNN-1D"] = cnn_results

    # LSTM
    lstm = EmotionLSTM(n_features=n_features, n_classes=n_classes)
    lstm_results = train_pytorch_model(
        lstm, X_train, y_train, X_test, y_test,
        "LSTM", device, n_epochs=100, lr=1e-3,
    )
    all_results["LSTM"] = lstm_results

    # Transformer (State-of-the-art attention model)
    from src.models.transformer_model import EEGTransformer
    transformer = EEGTransformer(
        n_features=n_features, n_classes=n_classes,
        d_model=128, n_heads=8, n_layers=4,
        dim_feedforward=256, dropout=0.3,
    )
    transformer_results = train_pytorch_model(
        transformer, X_train, y_train, X_test, y_test,
        "Transformer", device, n_epochs=100, lr=5e-4,
    )
    all_results["Transformer"] = transformer_results

    # ========================================
    # RESULTS SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 42)
    for name, res in all_results.items():
        print(f"{name:<20} {res['accuracy']:>10.4f} {res['f1']:>10.4f}")

    # ========================================
    # VISUALIZATIONS
    # ========================================
    print("\nGenerating visualizations...")

    # Training curves for deep models
    for name in ["MLP", "CNN-1D", "LSTM"]:
        if name in all_results and "history" in all_results[name]:
            plot_training_history(all_results[name]["history"], name)

    # Confusion matrices for all models
    for name, res in all_results.items():
        plot_confusion_matrix(
            y_test, res["preds"],
            class_names=class_names,
            model_name=name,
        )

    # Model comparison charts
    comparison = {
        name: {"accuracy": res["accuracy"], "f1": res["f1"]}
        for name, res in all_results.items()
    }
    plot_model_comparison(comparison, metric="accuracy")
    plot_model_comparison(comparison, metric="f1")

    print("\nAll done! Check results/figures/ for visualizations.")
    print("Models saved to results/models/")


if __name__ == "__main__":
    main()
