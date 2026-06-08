
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs
import sys
import glob
import numpy as np
from tf_keras.models import load_model

# Always resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def initialize():
    with open(os.path.join(BASE_DIR, "dataset", "header.txt"), "r") as file:
        headers = file.readlines()
        headers = [header.strip() for header in headers]
        headers = list(filter(None, headers))
        return headers


def extract_imports_from_exe(filepath):
    """Extract import function names directly from a PE32 .exe or .dll file."""
    import pefile
    functions = []
    try:
        pe = pefile.PE(filepath)
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for lib in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in lib.imports:
                    if imp.name:
                        functions.append(imp.name.decode("utf-8", errors="ignore"))
    except Exception as e:
        print(f"Error parsing PE file: {e}")
    return functions


def predict_from_functions(functions):
    """Predict malware category from a list of API function names."""
    global classifier
    row = [0 for i in range(len(headers))]
    if len(functions) > 5:
        for function in functions:
            if function[-1] == "W":
                function = function[:-1]
            elif function[-1] == "A":
                function = function[:-1]
            try:
                row[headers.index(function)] = 1
            except ValueError:
                pass
    probs = classifier.predict(np.array([row]), verbose=0)[0]
    pred = int(np.argmax(probs))
    confidence = float(probs[pred]) * 100
    return pred, confidence


def predict_txt(filepath):
    """Predict from a .txt file containing import function names."""
    with open(filepath, "r") as f:
        functions = [line.strip() for line in f if line.strip()]
    return predict_from_functions(functions)


def predict_exe(filepath):
    """Predict from a real .exe or .dll file."""
    functions = extract_imports_from_exe(filepath)
    if not functions:
        print(f"No imports found in {filepath}. The file may be packed or invalid.")
        return None, 0
    return predict_from_functions(functions)


if __name__ == "__main__":
    labels = {0: "Backdoor", 1: "Downloader", 2: "Keylogger", 3: "Miner",
              4: "Ransomware", 5: "Rouge Software", 6: "Trojan", 7: "Worm"}

    print("Loading model...")
    classifier = load_model(os.path.join(BASE_DIR, "adam.h5"))
    headers = initialize()
    print(f"Model loaded. {len(headers)} features.\n")

    # Ask the user for a file
    print("=" * 60)
    print("  MALWARE CLASSIFICATION SYSTEM")
    print("=" * 60)
    print("\nYou can provide:")
    print("  - A .exe or .dll file  (real malware sample)")
    print("  - A .txt file          (pre-extracted API imports)")
    print()

    filepath = input("Enter the file path (or press Enter to skip): ").strip()

    # Remove surrounding quotes if user drags & drops a file
    if filepath.startswith('"') and filepath.endswith('"'):
        filepath = filepath[1:-1]

    if filepath and os.path.isfile(filepath):
        filename = filepath.lower()
        if filename.endswith(".exe") or filename.endswith(".dll"):
            print(f"\nExtracting imports from: {os.path.basename(filepath)}")
            result, confidence = predict_exe(filepath)
        elif filename.endswith(".txt"):
            print(f"\nClassifying: {os.path.basename(filepath)}")
            result, confidence = predict_txt(filepath)
        else:
            print("\nUnsupported file type. Only .exe, .dll, or .txt files are accepted.")
            result = None

        if result is not None:
            print(f"\n  Result:     [{result}] {labels[result]}")
            print(f"  Confidence: {confidence:.1f}%\n")
    else:
        if filepath:
            print(f"\nFile not found: {filepath}")

        # Ask whether to run test samples or exit
        choice = input("\nWould you like to run test samples? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            print("\nRunning test samples from dataset/test/ ...\n")
            files = sorted(glob.glob(os.path.join(BASE_DIR, "dataset", "test", "*.txt")))
            if not files:
                print("No test files found in dataset/test/ folder.")
            else:
                for file in files:
                    result, confidence = predict_txt(file)
                    print(f"  {os.path.basename(file):30s} -->  [{result}] {labels.get(result, 'Unknown'):18s} ({confidence:.1f}%)")
                print(f"\n  Total: {len(files)} files classified.")
        else:
            print("\nExiting. Goodbye!")
