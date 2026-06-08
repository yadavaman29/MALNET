"""
DATA PIPELINE v3: Polymorphic Augmentation + 3-Way Split
==========================================================
This script takes the original 387 malware samples and:
1. Holds back 20% (77 samples) as a COMPLETELY UNSEEN test set
2. Uses the remaining 80% (310 samples) as the seed for augmentation
3. Generates 4,500 total augmented samples via Evasion Simulation
4. Splits those 4,500 augmented samples 80/20:
   - 3,600 for Training
   -   900 for Validation
5. Saves: augmented_train.csv, validation.csv, test_unseen.csv
"""

import os
import csv
import random
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

ORIGINAL_CSV = os.path.join(DATASET_DIR, "dataset.csv")
TRAIN_CSV = os.path.join(DATASET_DIR, "augmented_train.csv")
VALIDATION_CSV = os.path.join(DATASET_DIR, "validation.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test_unseen.csv")

SEED = 42
TARGET_AUGMENTED_TOTAL = 4500   # Total augmented pool
TRAIN_RATIO = 0.80              # 80% of augmented -> train (3600)
VAL_RATIO = 0.20                # 20% of augmented -> val   (900)
TEST_HOLDOUT_RATIO = 0.20       # 20% of original real data -> unseen test

LABELS = {
    0: "Backdoor", 1: "Downloader", 2: "Keylogger", 3: "Miner",
    4: "Ransomware", 5: "Rouge Software", 6: "Trojan", 7: "Worm"
}


def load_dataset(filepath):
    """Load CSV and return headers (feature names) and data rows."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            features = [int(x) for x in row[:-1]]
            label = int(row[-1])
            data.append((features, label))
    return headers[:-1], data


def stratified_split(data, split_ratio, seed):
    """
    Split data into two sets, maintaining class proportions.
    Even classes with very few samples (Miner=4) get at least 1 in the smaller set.
    """
    random.seed(seed)
    by_class = {}
    for features, label in data:
        by_class.setdefault(label, []).append((features, label))

    set_a, set_b = [], []
    for label, samples in sorted(by_class.items()):
        random.shuffle(samples)
        n_b = max(1, int(len(samples) * split_ratio))
        set_b.extend(samples[:n_b])
        set_a.extend(samples[n_b:])

    random.shuffle(set_a)
    random.shuffle(set_b)
    return set_a, set_b


def polymorphic_augment(sample_features, mutation_rate=0.10):
    """
    Evasion Simulation:
    Create a polymorphic variant by randomly dropping 1s to 0s.
    This simulates malware packers/obfuscators hiding API calls,
    forcing the neural network to learn from partial feature signatures.
    """
    augmented = sample_features.copy()
    existing_api_indices = [i for i, val in enumerate(augmented) if val == 1]

    if not existing_api_indices:
        return augmented

    n_mutations = max(1, int(len(existing_api_indices) * mutation_rate))
    indices_to_hide = random.sample(existing_api_indices, min(n_mutations, len(existing_api_indices)))
    for idx in indices_to_hide:
        augmented[idx] = 0

    return augmented


def augment_dataset(seed_data, target_size, seed):
    """
    Expand the seed data to target_size using polymorphic augmentation.
    Ensures balanced class representation.
    """
    random.seed(seed)
    np.random.seed(seed)

    by_class = {}
    for features, label in seed_data:
        by_class.setdefault(label, []).append(features)

    n_classes = len(by_class)
    target_per_class = target_size // n_classes

    augmented_data = []

    for label in sorted(by_class.keys()):
        originals = by_class[label]
        n_originals = len(originals)

        # Always include all original seed samples first
        for feat in originals:
            augmented_data.append((feat, label))

        # Generate polymorphic variants to reach target
        n_needed = target_per_class - n_originals
        for i in range(max(0, n_needed)):
            base = originals[i % n_originals]
            # Vary mutation rate between 5% and 25% for maximum diversity
            rate = random.uniform(0.05, 0.25)
            mutated = polymorphic_augment(base, mutation_rate=rate)
            augmented_data.append((mutated, label))

    random.shuffle(augmented_data)
    return augmented_data


def save_csv(filepath, headers, data):
    """Save dataset to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers + ["RESULT"])
        for features, label in data:
            writer.writerow(features + [label])


def main():
    print("=" * 60)
    print("  MALTOOL Data Pipeline v3")
    print("  3-Way Split: Train / Validation / Unseen Test")
    print("=" * 60)

    # Step 1: Load original dataset
    print("\n[1/5] Loading original dataset...")
    headers, data = load_dataset(ORIGINAL_CSV)
    print(f"  Loaded {len(data)} samples with {len(headers)} features")

    original_dist = Counter(label for _, label in data)
    print(f"  Original class distribution:")
    for label_id in sorted(original_dist):
        print(f"    [{label_id}] {LABELS[label_id]:18s}: {original_dist[label_id]:4d} samples")

    # Step 2: Hold back 20% of REAL data as completely unseen test set
    print(f"\n[2/5] Holding back 20% of real data as UNSEEN test set...")
    seed_data, test_data = stratified_split(data, TEST_HOLDOUT_RATIO, SEED)
    print(f"  Augmentation seed: {len(seed_data)} real samples")
    print(f"  Unseen test set:   {len(test_data)} real samples (NEVER TOUCHED)")

    test_dist = Counter(label for _, label in test_data)
    print(f"  Test set distribution:")
    for label_id in sorted(test_dist):
        print(f"    [{label_id}] {LABELS[label_id]:18s}: {test_dist[label_id]:4d} samples")

    # Step 3: Augment the seed data to 4,500 total samples
    print(f"\n[3/5] Augmenting seed data to {TARGET_AUGMENTED_TOTAL} samples...")
    augmented_pool = augment_dataset(seed_data, TARGET_AUGMENTED_TOTAL, SEED)
    print(f"  Augmented pool: {len(augmented_pool)} samples")

    aug_dist = Counter(label for _, label in augmented_pool)
    print(f"  Augmented class distribution:")
    for label_id in sorted(aug_dist):
        print(f"    [{label_id}] {LABELS[label_id]:18s}: {aug_dist[label_id]:4d} samples")

    # Step 4: Split the augmented pool 80/20 into train/val
    print(f"\n[4/5] Splitting augmented pool 80/20 for train/validation...")
    train_data, val_data = stratified_split(augmented_pool, VAL_RATIO, SEED + 1)
    print(f"  Training set:   {len(train_data)} augmented samples")
    print(f"  Validation set: {len(val_data)} augmented samples")

    train_dist = Counter(label for _, label in train_data)
    val_dist = Counter(label for _, label in val_data)
    print(f"\n  Training distribution:")
    for label_id in sorted(train_dist):
        print(f"    [{label_id}] {LABELS[label_id]:18s}: {train_dist[label_id]:4d}")
    print(f"\n  Validation distribution:")
    for label_id in sorted(val_dist):
        print(f"    [{label_id}] {LABELS[label_id]:18s}: {val_dist[label_id]:4d}")

    # Step 5: Save all three datasets
    print(f"\n[5/5] Saving datasets...")
    save_csv(TRAIN_CSV, headers, train_data)
    save_csv(VALIDATION_CSV, headers, val_data)
    save_csv(TEST_CSV, headers, test_data)
    print(f"  Training:   {TRAIN_CSV} ({len(train_data)} samples)")
    print(f"  Validation: {VALIDATION_CSV} ({len(val_data)} samples)")
    print(f"  Test:       {TEST_CSV} ({len(test_data)} samples)")

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Training:   {len(train_data):5d} augmented samples (for model.fit)")
    print(f"  Validation: {len(val_data):5d} augmented samples (for model.fit validation)")
    print(f"  Test:       {len(test_data):5d} REAL unseen samples (final accuracy proof)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
