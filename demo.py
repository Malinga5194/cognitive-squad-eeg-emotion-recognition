"""
LIVE DEMO SCRIPT
EEG-Based Cognitive State Classification - Cognitive Squad

Run this during your presentation to demonstrate the system live.
Usage: python demo.py

This script will:
1. Load the dataset
2. Show dataset statistics
3. Train a quick model (Random Forest)
4. Make predictions on test samples
5. Show results with visualizations
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def pause(msg="Press Enter to continue..."):
    input(f"\n>>> {msg}")


def main():
    print_header("COGNITIVE SQUAD - LIVE DEMO")
    print("  EEG-Based Emotion Recognition System")
    print("  MSc in Information Technology | SLIIT")
    pause("Press Enter to start the demo...")

    # ===== STEP 1: LOAD DATA =====
    print_header("STEP 1: Loading EEG Dataset")
    print("Loading emotions.csv from Muse EEG headband data...")
    time.sleep(1)

    df = pd.read_csv("data/raw/emotions.csv")
    print(f"  Samples loaded:  {df.shape[0]}")
    print(f"  Features/sample: {df.shape[1] - 1}")
    print(f"  Emotion classes: {df['label'].nunique()}")
    print(f"\n  Class distribution:")
    for label, count in df["label"].value_counts().items():
        bar = "#" * (count // 15)
        print(f"    {label:>10s}: {count} samples  {bar}")

    pause("Press Enter to preprocess data...")

    # ===== STEP 2: PREPROCESS =====
    print_header("STEP 2: Preprocessing")

    X = df.drop("label", axis=1).values.astype(np.float32)
    X = np.nan_to_num(X)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Z-score normalization: DONE")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")
    print(f"  Features:  {X_train.shape[1]}")

    pause("Press Enter to train the model...")

    # ===== STEP 3: TRAIN =====
    print_header("STEP 3: Training Random Forest Model")
    print("  Training Random Forest (300 trees)...")

    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=30,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"  Training complete in {train_time:.2f} seconds!")

    pause("Press Enter to see results...")

    # ===== STEP 4: EVALUATE =====
    print_header("STEP 4: Evaluation Results")

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"  ACCURACY: {acc:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    print("  Confusion Matrix:")
    print(f"  {'':>12s} {'NEGATIVE':>10s} {'NEUTRAL':>10s} {'POSITIVE':>10s}")
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i][j]:>10d}" for j in range(3))
        print(f"  {name:>12s} {row}")

    pause("Press Enter for live predictions...")

    # ===== STEP 5: LIVE PREDICTIONS =====
    print_header("STEP 5: Live Predictions on Random Samples")

    np.random.seed(int(time.time()) % 1000)
    indices = np.random.choice(len(X_test), 10, replace=False)

    print(f"  {'#':>3s}  {'Actual':>10s}  {'Predicted':>10s}  {'Match':>6s}")
    print(f"  {'-'*40}")

    correct = 0
    for i, idx in enumerate(indices):
        actual = class_names[y_test[idx]]
        predicted = class_names[preds[idx]]
        match = "YES" if actual == predicted else "NO"
        symbol = "+" if match == "YES" else "X"
        correct += 1 if match == "YES" else 0
        print(f"  {i+1:>3d}  {actual:>10s}  {predicted:>10s}  {symbol:>6s}")

    print(f"\n  Live accuracy: {correct}/10 = {correct*10}%")

    # ===== STEP 6: FEATURE IMPORTANCE =====
    pause("Press Enter to see top features...")
    print_header("STEP 6: Top 10 Most Important Features")

    feature_names = df.drop("label", axis=1).columns.tolist()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]

    for rank, idx in enumerate(top_idx):
        bar = "#" * int(importances[idx] * 500)
        print(f"  {rank+1:>2d}. {feature_names[idx]:<25s} {importances[idx]:.4f}  {bar}")

    # ===== DONE =====
    print_header("DEMO COMPLETE!")
    print(f"  Model: Random Forest (300 trees)")
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Dataset: 2,132 EEG samples, 3 emotion classes")
    print(f"\n  Thank you! - Cognitive Squad")
    print()


if __name__ == "__main__":
    main()
