import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from flask import Flask, request, jsonify, render_template
from tf_keras.models import load_model
import pefile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

LABELS = {
    0: ("Backdoor",      "Provides remote attacker access to the victim's machine.",           "#e74c3c"),
    1: ("Downloader",    "Downloads and executes other malicious payloads.",                   "#e67e22"),
    2: ("Keylogger",     "Logs keystrokes to steal credentials and sensitive data.",           "#9b59b6"),
    3: ("Miner",         "Uses victim's CPU/GPU to mine cryptocurrency for the attacker.",     "#f39c12"),
    4: ("Ransomware",    "Encrypts files and demands payment for decryption.",                 "#c0392b"),
    5: ("Rouge Software","Tricks users into buying fake/harmful security software.",           "#16a085"),
    6: ("Trojan",        "Disguises itself as legitimate software while doing malicious acts.","#2980b9"),
    7: ("Worm",          "Self-replicates across networks and systems.",                       "#27ae60"),
}

# Load model and headers once at startup
print("[*] Loading model...")
classifier = load_model(os.path.join(BASE_DIR, "adam.h5"))
print("[*] Loading headers...")
with open(os.path.join(BASE_DIR, "dataset", "header.txt"), "r") as f:
    HEADERS = [h.strip() for h in f if h.strip()]
print(f"[*] Ready! {len(HEADERS)} features loaded.")


def extract_imports_from_exe(file_bytes):
    """Extract import function names from a PE32 binary using pefile."""
    imports = []
    try:
        pe = pefile.PE(data=file_bytes)
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for lib in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in lib.imports:
                    if imp.name:
                        name = imp.name.decode("utf-8", errors="ignore")
                        imports.append(name)
    except Exception as e:
        raise ValueError(f"Failed to parse PE file: {str(e)}")
    return imports


def extract_imports_from_txt(text):
    """Extract function names from an already-extracted .txt file."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_feature_vector(functions):
    """Convert a list of API function names into a one-hot encoded feature vector."""
    row = [0] * len(HEADERS)
    matched = 0
    for fn in functions:
        # Strip Windows A/W suffix (e.g. CreateFileW → CreateFile)
        stem = fn
        if len(fn) > 1 and fn[-1] == "W":
            stem = fn[:-1]
        elif len(fn) > 1 and fn[-1] == "A":
            stem = fn[:-1]
        try:
            row[HEADERS.index(stem)] = 1
            matched += 1
        except ValueError:
            pass
    return row, matched


def run_prediction(functions):
    if len(functions) <= 5:
        raise ValueError("Too few import functions found (≤5). The file may be packed or invalid.")
    row, matched = build_feature_vector(functions)
    probs = classifier.predict(np.array([row]), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    all_probs = {LABELS[i][0]: round(float(probs[i]) * 100, 2) for i in range(8)}
    label, desc, color = LABELS[pred_idx]
    return {
        "label": label,
        "label_id": pred_idx,
        "description": desc,
        "confidence": round(confidence, 2),
        "color": color,
        "all_probabilities": all_probs,
        "total_imports": len(functions),
        "matched_features": matched,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    filename = f.filename.lower()

    try:
        if filename.endswith(".exe") or filename.endswith(".dll"):
            file_bytes = f.read()
            functions = extract_imports_from_exe(file_bytes)
        elif filename.endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
            functions = extract_imports_from_txt(text)
        else:
            return jsonify({"error": "Unsupported file type. Please upload a .exe, .dll, or .txt file."}), 400

        if not functions:
            return jsonify({"error": "No import functions found in this file."}), 400

        result = run_prediction(functions)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/test-samples", methods=["POST"])
def test_samples():
    """Run prediction on all test samples in dataset/test/ folder."""
    import glob
    test_dir = os.path.join(BASE_DIR, "dataset", "test")
    files = sorted(glob.glob(os.path.join(test_dir, "*.txt")))
    if not files:
        return jsonify({"error": "No test samples found."}), 404

    results = []
    for filepath in files:
        try:
            with open(filepath, "r") as f:
                text = f.read()
            functions = extract_imports_from_txt(text)
            if functions and len(functions) > 5:
                result = run_prediction(functions)
                result["filename"] = os.path.basename(filepath)
                results.append(result)
        except Exception:
            pass
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
