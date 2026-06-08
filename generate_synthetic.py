"""
SYNTHETIC GENERATOR: Simulates 5,000 extracted malware .txt files
==================================================================
This script generates purely synthetic feature vectors (lists of APIs)
and saves them as individual .txt files.

It allows you to test the deployment scale of your web app or
your prediction pipeline during your college presentation without
needing thousands of real malicious `.exe` files.
"""

import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADER_PATH = os.path.join(BASE_DIR, "dataset", "header.txt")
STRESS_DIR = os.path.join(BASE_DIR, "dataset", "stress_test")

LABELS = [
    "Backdoor", "Downloader", "Keylogger", "Miner",
    "Ransomware", "Rouge_Software", "Trojan", "Worm"
]

def load_headers():
    with open(HEADER_PATH, "r") as f:
        return [h.strip() for h in f if h.strip()]

def generate_synthetic_samples(num_samples=5000):
    print("=" * 60)
    print("  MALTOOL Synthetic Generator")
    print(f"  Generating {num_samples} files...")
    print("=" * 60)

    # 1. Create directory
    if not os.path.exists(STRESS_DIR):
        os.makedirs(STRESS_DIR)
        print(f"[*] Created directory: {STRESS_DIR}")

    # 2. Load API features
    try:
        features = load_headers()
        print(f"[*] Loaded {len(features)} possible Windows API functions")
    except FileNotFoundError:
        print("[!] Error: dataset/header.txt not found!")
        return

    # 3. Generate files
    for i in range(1, num_samples + 1):
        # Pick a random malware family name for the filename
        family = random.choice(LABELS)
        filename = f"synthetic_{family.lower()}_{i:04d}.txt"
        
        # A typical sample might import anywhere from 10 to 150 APIs
        num_imports = random.randint(10, 150)
        
        # Randomly select APIs to match the size
        sample_apis = random.sample(features, num_imports)

        # Write to file
        filepath = os.path.join(STRESS_DIR, filename)
        with open(filepath, "w") as f:
            for api in sample_apis:
                f.write(api + "\n")

        if i % 1000 == 0:
            print(f"  [+] Generated {i}/{num_samples} files...")

    print(f"\n[*] SUCCESSFULLY GENERATED {num_samples} FILES")
    print(f"[*] Location: {STRESS_DIR}")

if __name__ == "__main__":
    generate_synthetic_samples()
