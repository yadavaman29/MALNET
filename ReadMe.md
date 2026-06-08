# MALNET: Deep Learning Malware Classification System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)](https://flask.palletsprojects.com/)

**MALNET** is an automated deep learning system that classifies Windows Portable Executable (PE32) malware into eight distinct families without ever executing the malicious binary. The core objective is to demonstrate that static features drawn from the Import Address Table (IAT) of a PE file carry sufficient discriminative information for reliable, execution-free classification.

This repository contains the source code, pre-trained models, and documentation for the project, which achieves approximately **96% accuracy** on unseen real-world malware samples.

---

## 🌟 Key Features

- **Execution-Free Static Analysis**: Extracts Windows API function calls from the import tables of PE32 binaries, eliminating the need to run malware in a sandbox.
- **Deep Neural Network Classifier**: Built using Keras/TensorFlow, featuring a fully connected architecture with ReLU activation, graduated dropout, and L2 weight regularization.
- **8 Malware Families Classified**: Supports detecting Backdoor, Downloader, Keylogger, Miner, Ransomware, Rogue Software, Trojan, and Worm.
- **Polymorphic Data Augmentation**: Simulates real-world evasion tactics to expand the dataset from 387 verified real-world samples to 4,500 balanced samples.
- **Web Interface & CLI Utility**: Includes a Flask-based web application and CLI tools (`predict.py`) for real-time predictions with confidence scores.
- **Synthetic Data Generation**: Allows stress-testing and demonstrating the model at scale without the risk of handling live malware.

---

## 🛠️ Project Structure

```text
.
├── app.py                             # Flask web application for uploading and classifying samples
├── predict.py                         # CLI script to predict malware class from a PE file
├── retrain_model.py                   # Script to train/retrain the Deep Neural Network
├── data_pipeline.py                   # Data augmentation and pipeline simulation
├── generate_synthetic.py              # Generates synthetic API features for stress testing
├── extract_apis.py                    # Utility to extract imports statically using pefile
├── adam.h5                            # Pre-trained deep learning model
├── Malware_Classification_Optimized.ipynb # Jupyter Notebook detailing the model optimization
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Standard Python gitignore
├── dataset/                           # Contains the feature sets
├── static/                            # Web application static assets (CSS, images)
├── templates/                         # Web application HTML templates
```

---

## 🚀 Getting Started

### Prerequisites

You need Python 3.6 or newer to run this project. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-username/Malware-Classification-and-Labelling.git
cd Malware-Classification-and-Labelling

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required dependencies
pip install -r requirements.txt
```

### Usage

**1. Command Line Interface (CLI)**
To classify a specific Windows executable (`.exe` or `.dll`) directly from your terminal:

```bash
python predict.py path/to/your/suspicious_file.exe
```

**2. Web Application**
To start the Flask web server with a user-friendly UI:

```bash
python app.py
```
Then, open your browser and navigate to `http://127.0.0.1:5000`. You can upload executables or pre-extracted API text files to get a real-time prediction.

**3. Retraining the Model**
If you have new data or want to build the model from scratch using the existing dataset:
```bash
python retrain_model.py
```

**4. Generating Synthetic Data for Testing**
To test the pipeline or web app without handling real malware samples:
```bash
python generate_synthetic.py
```

---

## 🤝 Sample Collection and Contribution

Most samples are collected from various GitHub repositories where the malware has been classified already. Thanks to [VirusSign](https://www.virussign.com/) and [VirusShare](https://virusshare.com/) for providing access to a huge range of malware collections.

**We are looking for unique Worm samples!** Variants like polymorphic worms (e.g., Allaple) create data duplication since they share similar API patterns despite different file hashes. If you have unique collections of worms, downloaders, keyloggers, or crypto-miners, kindly contribute by creating a pull request with the **imports txt file alone (No EXE or live malware please)**.

---

## 📖 Detailed Documentation & Methodology

*The following sections outline the detailed methodology, algorithms, and architectures used in this research.*

<details>
<summary>Click here to expand the detailed methodology.</summary>

### 1. Abstract
The rapid proliferation of malware poses a growing threat to digital infrastructure worldwide. With millions of new malicious samples surfacing each day, manually examining and categorising every specimen through traditional reverse-engineering workflows is neither practical nor scalable. This minor project, titled **MALNET**, presents an automated system that classifies Windows Portable Executable (PE32) malware into eight distinct families without ever executing the malicious binary. The core objective is to demonstrate that static features drawn from the Import Address Table of a PE file carry sufficient discriminative information for reliable, execution-free classification.

### 2. Dataset Preparation & Import Extraction

A C++ based extractor (and a Python equivalent `extract_apis.py`) scans PE files, extracting imports to identify features. MD5 hashing prevents data duplication. Packed malware (e.g., UPX) is filtered out or handled separately.

**Algorithm 1: Import Extraction**
```text
for malware in directories
    if malware == scanned: skip
    else if malware == packed: append packed.txt -> malware_hash; skip
    imports = get_all_imports(malware)
    write malware_hash.txt -> imports
    append frequency.txt -> imports
    append scanned_file.txt -> malware_hash
```

**Algorithm 2: Compiling the Dataset**
```text
import_list = []
for frequency_file in directory:
    imports = get_imports(frequency_file)
    import_list.append(imports)
remove_duplicates(import_list)
remove_unwanted_import(import_list)
sort(import_list)
create_column_headers(import_list)

for malware_hash_file in directory:
    imports = get_imports(malware_hash)
    row = init_zeros(length -> header) + malware_type
    for function in imports:
        row[header.index(function)] = 1
    add_row(row)
```

### 3. Data Augmentation and Pipeline

To improve model robustness and prevent overfitting, the `data_pipeline.py` employs **Polymorphic Data Augmentation**, simulating malicious evasion techniques (such as packers or obfuscators hiding API calls) by randomly dropping subsets of recorded API features to 0.

1. **3-Way Data Split**: 20% of the real malware samples are entirely withheld as an **Unseen Test Set**.
2. **Augmentation**: The remaining 80% seed dataset is expanded to **4,500 total augmented samples**, perfectly balanced.
3. **Training & Validation**: The expanded pool is split 80/20 for model training and validation.

### 4. Deep Learning Model Architecture

The network architecture has been meticulously optimized for generalization:
1. **Input Layer**: Takes the one-hot encoded feature vector.
2. **Hidden Layer 1**: 256 units (ReLU + L2 Regularization), followed by 30% Dropout.
3. **Hidden Layer 2**: 128 units (ReLU + L2 Regularization), followed by 40% Dropout.
4. **Hidden Layer 3**: 64 units (ReLU + L2 Regularization), followed by 50% Dropout.
5. **Output Layer**: 8 units corresponding to the malware classes, utilizing SoftMax activation.

The model is optimized using the Adam optimizer to minimize Sparse Categorical Crossentropy loss with Early Stopping, achieving a rigorous **96% accuracy** on the held-out unseen real dataset.

### 5. Notable Malware Family API Patterns
- **Backdoors**: `CreateFile`, `GetProcAddress`, `GetTickCount`, `VirtualAlloc`.
- **Downloaders**: `CreateFile`, `WriteFile`, `Sleep`, `GetTickCount`.
- **Keyloggers**: `ReadFile`, `WriteFile` (keyboard logging).
- **Miners**: Multi-threading (`GetCurrentThread`, `ResumeThread`) and Networking (`InternetReadFile`).
- **Ransomwares**: `ReadFile`, `WriteFile` (bulk file modifications/encryption).
- **Trojans/Worms**: Registry manipulation, `DeleteFile`, `TerminateProcess`.

*(For complete frequency distribution graphs and analysis, see the images in the `static/` directory)*

</details>

---

## 📚 References

1. Virus total statistics: https://www.virustotal.com/en/statistics/ 
2. J.-M. Roberts. Virus Share. https://virusshare.com/ 
3. K. Rieck, P. Trinius, C. Willems , T. Holz. Automatic Analysis of Malware Behaviour using Machine Learning.
4. B. Kolosnjaji, A. Zarras, G. Webster, and C. Eckert. Deep Learning for Classification of Malware System Call Sequences.
5. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 2014.
6. Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
7. Sikorski, Michael, and Andrew Honig. Practical malware analysis: the hands-on guide to dissecting malicious software. No Starch Press, 2012.

---
*Note: This repository is created for educational and research purposes only.*
