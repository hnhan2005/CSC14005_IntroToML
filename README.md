# Vietnamese Handwritten OCR - Introduction to Machine Learning

## Course Information

| Item | Details |
|---|---|
| **Course** | CSC14005 - Introduction to Machine Learning |
| **Class** | ML 23KHDL1 |
| **Group** | 6 |

### Team Members

| Student ID | Full Name |
|---|---|
| 23127102 | Le Quang Phuc |
| 23127212 | Nguyen Quang Dang Khoa |
| 23127241 | Doan Thanh Phat |
| 23127332 | Tran Tien Cuong |
| 23127442 | Tram Huu Nhan |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [Data Exploration and Preprocessing](#1-data-exploration-and-preprocessing)
  - [Data Preparation](#2-data-preparation)
  - [Baseline Evaluation](#3-baseline-evaluation)
  - [Fine-tuning](#4-fine-tuning)
- [Models](#models)
- [Evaluation Metric](#evaluation-metric)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [How to Reproduce](#how-to-reproduce)
- [Documentation](#documentation)

---

## Project Overview

This project tackles the problem of **Vietnamese Optical Character Recognition (OCR)** on handwritten text. The goal is to evaluate and compare the performance of multiple OCR models on a Vietnamese handwriting dataset, and then improve recognition accuracy through fine-tuning.

The project follows a structured machine learning workflow:

1. **Data Exploration & Preprocessing** -- Inspect, clean, and analyze the raw dataset.
2. **Data Preparation** -- Stratified sampling and splitting into Train / Valid / Test sets.
3. **Baseline Evaluation** -- Benchmark three pre-trained OCR models without any adaptation.
4. **Fine-tuning** -- Apply LoRA-based parameter-efficient fine-tuning to the best-performing model and measure improvement.

---

## Dataset

The dataset is composed of two main sources:

| Source | Description | Granularity |
|---|---|---|
| **UIT-HWDB** | Public Vietnamese handwriting dataset from UIT | Word, Line, Paragraph |
| **Manual** | Self-collected handwriting samples by team members (5 contributors, ~100 images each) | Mixed |

### Dataset Statistics

| Source | Samples |
|---|---|
| UIT_HWDB_word | 110,483 |
| UIT_HWDB_line | 7,228 |
| UIT_HWDB_paragraph | 1,144 |
| Manual (5 members) | ~494 |
| **Total** | **~119,349** |

### Data Split

From the total pool, **100,000 samples** were randomly selected with **stratified proportional sampling** (preserving the ratio of each source) and divided as follows:

| Split | Samples | Proportion |
|---|---|---|
| Train | 70,000 | 70% |
| Validation | 15,000 | 15% |
| Test | 15,000 | 15% |

Each split is organized into subfolders of 100 images, with a corresponding `label.json` file per subfolder.

---

## Project Structure

```
CSC14005_IntroToML/
|
|-- docs/
|   |-- Proposal.pdf                  # Project proposal document
|   |-- Data report.pdf               # Data analysis report
|
|-- notebooks/
|   |-- exploration_data.ipynb        # EDA: cleaning, statistics, visualization
|   |-- data_preparation.ipynb        # Stratified split into Train/Valid/Test
|   |-- baseline_tesseract5x.ipynb    # Baseline: Tesseract 5.x
|   |-- baseline_PPOCR.ipynb          # Baseline: PaddleOCR
|   |-- baseline_GLMOCR.ipynb         # Baseline: GLM-OCR
|   |-- finetune_GLMOCR.ipynb         # Fine-tuning GLM-OCR with LoRA
|
|-- baseline.csv                      # Baseline evaluation results (subset)
|-- finetune.csv                      # Fine-tuned model evaluation results
|-- baseline_Tesseract_report.csv     # Full error report: Tesseract
|-- baseline_PPOCR_report.csv         # Full error report: PaddleOCR
|-- baseline_GLMOCR_report.csv        # Full error report: GLM-OCR
|
|-- README.md
```

---

## Pipeline

### 1. Data Exploration and Preprocessing

**Notebook:** `notebooks/exploration_data.ipynb`

- **Data Cleaning:** An `OCRDatasetCleaner` class was built to ensure dataset integrity by:
  - Removing images without corresponding labels
  - Removing labels without corresponding images
  - Deleting corrupted or incomplete subfolders
- **Exploratory Data Analysis (EDA):**
  - Statistics on character/word frequency, text length distribution
  - Image resolution, aspect ratio, file size, brightness, and sharpness analysis
  - Word clouds and distribution visualizations
- **Data Archiving:** Cleaned data is packaged and stored as `Data_Clean.zip` (~1.93 GB)

### 2. Data Preparation

**Notebook:** `notebooks/data_preparation.ipynb`

- **Stratified Proportional Sampling:** Data from all four sources (Manual, UIT Word, UIT Line, UIT Paragraph) is shuffled and split proportionally so each subset (Train/Valid/Test) maintains the same data-source distribution as the full dataset.
- **Output Format:** Each split is organized into subfolders of 100 images with `label.json` files, then compressed into `processed_data.zip` for use in subsequent steps.

### 3. Baseline Evaluation

Three pre-trained OCR models were benchmarked **without any fine-tuning** on the full test set (15,000 images):

#### Tesseract 5.x

**Notebook:** `notebooks/baseline_tesseract5x.ipynb`

- Traditional OCR engine with Vietnamese language pack (`Vietnamese.traineddata`)
- Adaptive Page Segmentation Mode (PSM) based on image aspect ratio:
  - Aspect ratio > 2.5 (wide images): `--psm 7` (single text line)
  - Aspect ratio <= 2.5 (block-shaped): `--psm 6` (uniform block of text)
- Used `pytesseract` as the Python interface

#### PaddleOCR (PPOCR)

**Notebook:** `notebooks/baseline_PPOCR.ipynb`

- Deep learning-based OCR framework with Vietnamese language support (`lang='vi'`)
- Detection and angle classification disabled (`det=False`, `use_angle_cls=False`) to isolate recognition performance on pre-cropped images
- Used `paddleocr` Python package (v2.7.3) with PaddlePaddle GPU backend

#### GLM-OCR

**Notebook:** `notebooks/baseline_GLMOCR.ipynb`

- Vision-Language model from Hugging Face (`zai-org/GLM-OCR`)
- Multimodal architecture combining a vision encoder with a language model
- Loaded in FP16 precision with `device_map="auto"` for GPU T4 compatibility
- Inference via structured chat-based prompt: `"Text Recognition:"`

### 4. Fine-tuning

**Notebook:** `notebooks/finetune_GLMOCR.ipynb`

The GLM-OCR model was selected for fine-tuning using **LoRA (Low-Rank Adaptation)** combined with **8-bit quantization** via the **LLaMA-Factory** framework.

#### Fine-tuning Configuration

| Parameter | Value |
|---|---|
| Method | LoRA (rank=8, target=all) |
| Quantization | 8-bit |
| Epochs | 3 |
| Batch Size | 2 |
| Gradient Accumulation | 4 (effective batch=8) |
| Learning Rate | 1e-4 |
| LR Scheduler | Cosine with warmup (ratio=0.1) |
| Max Sequence Length | 2,048 |
| Training Samples | 1,000 |
| Precision | BF16 |
| Framework | LLaMA-Factory v0.9.5 |

#### Fine-tuning Workflow

1. **Data Conversion:** Dataset converted to ShareGPT format (chat-style messages with image references)
2. **Dataset Registration:** Registered in LLaMA-Factory's `dataset_info.json`
3. **Training:** Executed via `llamafactory-cli train` with YAML configuration
4. **Checkpoint Management:** Supports resume from checkpoint, with automatic backup to Google Drive
5. **Evaluation:** Per-checkpoint CER evaluation on validation set, with loss curve visualization

---

## Models

| Model | Type | Size | Language Support |
|---|---|---|---|
| **Tesseract 5.x** | Traditional OCR (LSTM-based) | ~15 MB | Vietnamese (script model) |
| **PaddleOCR** | Deep Learning (CRNN + CTC) | ~10 MB | Vietnamese (pre-trained) |
| **GLM-OCR** | Vision-Language Model (Transformer) | ~2.65 GB | Multilingual |

---

## Evaluation Metric

**CER (Character Error Rate)** is used as the primary evaluation metric, based on Levenshtein distance:

```
CER = (S + D + I) / N
```

Where:
- **S** = Number of character substitutions
- **D** = Number of character deletions
- **I** = Number of character insertions
- **N** = Total characters in the ground truth

**Lower CER = Better performance** (CER = 0 means perfect recognition).

All text is normalized to **NFC Unicode** and lowercased before comparison to avoid encoding discrepancies common in Vietnamese text.

---

## Results

### Baseline Evaluation (Full Test Set -- 15,000 images)

| Model | Average CER | Error Count (/15,000) |
|---|---|---|
| **PaddleOCR** | **57.29%** | 13,837 |
| GLM-OCR | 90.40% | 12,599 |
| Tesseract 5.x | 98.59% | 14,741 |

### Fine-tuned GLM-OCR (Subset -- 200 images)

| Checkpoint | Average CER |
|---|---|
| Baseline (no fine-tune) | 85.00% |
| checkpoint-100 | 33.91% |
| checkpoint-189 | ~30% |
| checkpoint-200 | ~28% |
| checkpoint-300 | ~25% |
| checkpoint-375 | ~22% |

> Fine-tuning with LoRA reduced the CER of GLM-OCR from **85%** to approximately **22%** on the evaluation subset, demonstrating a significant improvement of over **60 percentage points**.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.12+ |
| **Environment** | Google Colab (GPU T4) |
| **Core Libraries** | PyTorch, Transformers, PaddlePaddle |
| **Fine-tuning** | PEFT (LoRA), BitsAndBytes, LLaMA-Factory |
| **OCR Engines** | Tesseract 5.x (pytesseract), PaddleOCR, GLM-OCR |
| **Evaluation** | Levenshtein, pandas |
| **Visualization** | Matplotlib, Seaborn, Plotly, WordCloud |
| **Data Processing** | NumPy, PIL/Pillow, JSON, pathlib |

---

## How to Reproduce

### Prerequisites

- Google Account with access to Google Colab (GPU runtime recommended)
- Dataset stored on Google Drive in the path: `IntroToML - OCR - data/`

### Step-by-step

1. **Data Exploration & Cleaning**
   ```
   Run: notebooks/exploration_data.ipynb
   ```
   Cleans the raw dataset and generates `Data_Clean.zip`.

2. **Data Preparation**
   ```
   Run: notebooks/data_preparation.ipynb
   ```
   Creates stratified Train/Valid/Test splits as `processed_data.zip`.

3. **Baseline Evaluation**
   ```
   Run (in any order):
     - notebooks/baseline_tesseract5x.ipynb
     - notebooks/baseline_PPOCR.ipynb
     - notebooks/baseline_GLMOCR.ipynb
   ```
   Generates error report CSVs for each model.

4. **Fine-tuning**
   ```
   Run: notebooks/finetune_GLMOCR.ipynb
   ```
   Fine-tunes GLM-OCR using LoRA and evaluates on validation set.

> **Note:** All notebooks are designed to run on Google Colab. They automatically mount Google Drive, extract data to local storage for I/O performance, and save results back to Drive.

---

## Documentation

| Document | Description |
|---|---|
| [Proposal.pdf](docs/Proposal.pdf) | Project proposal and objectives |
| [Data report.pdf](docs/Data%20report.pdf) | Data analysis and statistics report |
