"""
Training and Evaluation Utilities
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm


def get_device() -> torch.device:
    """Get the best available device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Returns:
        loss, accuracy, all_predictions, all_true_labels
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: torch.device | None = None,
    model_name: str = "model",
) -> dict:
    """
    Full training loop with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        device: Device to train on
        model_name: Name for saving/logging

    Returns:
        Dictionary with training history and best metrics
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"results/models/{model_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Load best model for final evaluation
    model.load_state_dict(
        torch.load(f"results/models/{model_name}_best.pth", weights_only=True)
    )

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["training_time"] = elapsed

    return history
