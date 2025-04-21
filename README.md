# ProtoCluster Knows Where the Bodies Are Buried (in the Protein)

## 🧬 Pipeline

This pipeline processes `.vtk`-based datasets, trains a multi-stage GNN model, and generates final predictions suitable for submission.

---

### 📖 Overview

The pipeline is structured into multiple stages:

1. **Installation**: Environment setup and dependency installation.
2. **Data Preparation**: Downloading and extracting `.tar.gz` archives containing the `.vtk` files.
3. **Stage 1 - Confusion Analysis**: Initial training and error identification.
4. **Stage 2 - Pretraining**: Retraining with refined labels.
5. **Stage 3 - Finetuning**: Final model training with pretrained weights.
6. **Inference & Submission**: Running inference on test data and saving results.

---

## ⚙️ Installation

**For macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**For Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📦 Data Download

A helper script is provided to download sample datasets. If on Windows, run it with **Git Bash**.

```bash
sh scripts/download_data.sh
```

---

## 🚀 Run the Pipeline

By default:
- `$DATA_PATH = ./sample_data`
- `$OUTPUT_PATH = ./output`

You may replace `sample_data` with a full dataset (e.g., `./data`) if available.

```bash
sh scripts/pipeline.sh
```