"""
RETRAIN MODEL v3: Train on Augmented Data, Validate, Test on Unseen Real Data
===============================================================================
Architecture (Regularized):
  Input (1858) -> Dense(256, ReLU, L2) -> Dropout(0.3)
              -> Dense(128, ReLU, L2) -> Dropout(0.4)
              -> Dense(64,  ReLU, L2) -> Dropout(0.5)
              -> Dense(8, Softmax) [Output]

Optimizer: Adam | Loss: Sparse Categorical Crossentropy
Early Stopping on val_loss with patience=20
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import shutil
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
from tf_keras.optimizers import Adam
from tf_keras.regularizers import l2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_CSV = os.path.join(DATASET_DIR, "augmented_train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "validation.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test_unseen.csv")
MODEL_PATH = os.path.join(BASE_DIR, "adam.h5")
BACKUP_PATH = os.path.join(BASE_DIR, "adam_backup_original.h5")

LABELS = {
    0: "Backdoor", 1: "Downloader", 2: "Keylogger", 3: "Miner",
    4: "Ransomware", 5: "Rouge Software", 6: "Trojan", 7: "Worm"
}


def load_csv(filepath):
    """Load CSV, return features (numpy array) and labels (numpy array)."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        features, labels = [], []
        for row in reader:
            features.append([int(x) for x in row[:-1]])
            labels.append(int(row[-1]))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_model(input_dim, num_classes=8):
    """Build a regularized DNN with graduated dropout (0.3 -> 0.4 -> 0.5)."""
    model = Sequential([
        Dense(256, activation="relu", kernel_regularizer=l2(0.01), input_dim=input_dim),
        Dropout(0.3),
        Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("=" * 60)
    print("  MALTOOL Model Retraining v3")
    print("  Train / Validate / Test (Unseen Real Data)")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading datasets...")
    X_train, y_train = load_csv(TRAIN_CSV)
    X_val, y_val = load_csv(VAL_CSV)
    X_test, y_test = load_csv(TEST_CSV)
    print(f"  Training:   {X_train.shape[0]} augmented samples")
    print(f"  Validation: {X_val.shape[0]} augmented samples")
    print(f"  Test:       {X_test.shape[0]} REAL unseen samples")

    # Build model
    print("\n[2/6] Building regularized DNN architecture...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # Backup old model
    print("\n[3/6] Backing up original model...")
    if os.path.exists(MODEL_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"  Backed up to: {BACKUP_PATH}")
    else:
        print(f"  Backup already exists or no model to backup.")

    # Train
    print("\n[4/6] Training (Early Stopping on val_loss, patience=20)...")
    checkpoint = ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", save_best_only=True,
        mode="min", verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Evaluate on validation
    print("\n[5/6] Evaluation on VALIDATION set (augmented)...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

    print(f"\n  Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Validation Loss:     {val_loss:.4f}")

    # Evaluate on UNSEEN test set (the real proof)
    print("\n[6/6] FINAL EVALUATION on UNSEEN REAL test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Training Accuracy:   {train_acc * 100:.2f}% (on 3,600 augmented)")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% (on 900 augmented)")
    print(f"  TEST Accuracy:       {test_acc * 100:.2f}% (on {X_test.shape[0]} REAL unseen)")
    print(f"  Test Loss:           {test_loss:.4f}")
    print(f"  Model saved to:      {MODEL_PATH}")
    print(f"{'=' * 60}")

    # Per-class accuracy on UNSEEN test set
    print("\n  Per-class TEST accuracy (REAL unseen data):")
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    for label_id in sorted(LABELS.keys()):
        mask = y_test == label_id
        if mask.sum() > 0:
            class_acc = (predictions[mask] == y_test[mask]).mean() * 100
            print(f"    [{label_id}] {LABELS[label_id]:18s}: {class_acc:6.1f}%  ({mask.sum()} samples)")
        else:
            print(f"    [{label_id}] {LABELS[label_id]:18s}:   N/A   (0 samples)")

    # Save training history
    history_path = os.path.join(DATASET_DIR, "training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for i in range(len(history.history["loss"])):
            writer.writerow([
                i + 1,
                round(history.history["loss"][i], 6),
                round(history.history["accuracy"][i], 6),
                round(history.history["val_loss"][i], 6),
                round(history.history["val_accuracy"][i], 6),
            ])
    print(f"\n  Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
