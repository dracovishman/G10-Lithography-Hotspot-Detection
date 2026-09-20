# G10 — Focal and Hard-Example Learning for Lithography Hotspot Detection on ICCAD-12

[![Course](https://img.shields.io/badge/Course-BEVD402L--AI%20%26%20ML%20for%20IC-blue.svg)](https://vit.ac.in)
[![Institution](https://img.shields.io/badge/Institution-VIT%20Chennai-orange.svg)](https://vit.ac.in)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

Official source code, experimental reproducibility suite, and benchmark evaluation for **Group 10 (G10)** — Digital Assignment II (DA-2):
* **Course:** AI and Machine Learning for IC Design (BEVD402L) | Slot: D1+TD1
* **Faculty Supervisor:** Dr. G. Lakshmi Priya
* **Institution:** School of Electronics Engineering (SENSE), Vellore Institute of Technology, Chennai

---

## 👥 Group 10 Team Members & Work Allocation

| Reg. No. | Student Name | Role / Key Module Contribution | Contribution % |
| :---: | :--- | :--- | :---: |
| **23BVD1062** | Vinayak Shreenivas Salunke | Dataset Preprocessing, BCE & Focal Loss Baseline Execution, Result Aggregation | **33.3%** |
| **23BVD1064** | Harsh Vardhan Singh | OHEM Architecture Formulation, IEEE Report Writing, Discussion & Limitation Analysis | **33.3%** |
| **23BVD1065** | Alavala Vishnu Koushik Reddy | Confusion Matrix Heatmaps, Comparative Plot Generation, Slide Deck Preparation | **33.3%** |

---

## 📌 Executive Summary & Research Context

Lithography hotspot detection identifies layout topology clips susceptible to manufacturing defects (open circuit line-end pinching or short circuit line bridging) during photolithographic semiconductor fabrication. In the industry-standard **ICCAD-12 benchmark suite**, non-hotspot patterns vastly outnumber hotspot patterns (e.g., Benchmark 5 test set contains only 41 hotspots out of 19,368 clips, i.e., $0.21\%$ hotspot prevalence).

### Core Research Question (RQ)
> *"Can a learning strategy emphasizing difficult and minority samples improve lithography hotspot detection reliability under extreme class imbalance without adding CNN model complexity?"*

### Key Finding
**YES.** Replacing standard Binary Cross-Entropy (BCE) loss with **Focal Loss** ($\alpha=0.75, \gamma=2.0$) increases the five-benchmark arithmetic-mean **Balanced Accuracy from 82.78% to 93.02%** (+10.24 percentage points) and mean **Hotspot Recall from 86.73% to 96.45%** (+9.72 percentage points) while maintaining a lightweight CNN baseline of only **12,873 parameters** and **~6.3 ms/clip** inference latency.

---

## 📊 Summary Performance Comparison Across ICCAD-12

### 1. Overall Five-Benchmark Arithmetic Mean Metrics

| Primary Evaluation Metric | BCE Baseline (Mean) | Proposed Focal Loss (Mean) | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy (Primary)** | **82.78%** | **93.02%** | **+10.24 pp** |
| **Recall / Sensitivity** | **86.73%** | **96.45%** | **+9.72 pp** |
| **Specificity** | **78.84%** | **89.58%** | **+10.74 pp** |
| **Precision** | **27.07%** | **31.96%** | **+4.89 pp** |
| **F1-Score** | **36.88%** | **42.58%** | **+5.70 pp** |

### 2. Benchmark-Wise Test Performance (B1 to B5)

| Benchmark | Training Method | Balanced Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **B1** | BCE Baseline | 83.16% | 12.94% | 98.23% | 68.09% | 22.87% |
| **B1** | **Focal Loss (α=0.75, γ=2)** | **87.28%** | **17.77%** | 96.02% | **78.54%** | **29.99%** |
| **B2** | BCE Baseline | **98.92%** | 46.96% | **99.20%** | 98.65% | 63.74% |
| **B2** | **Focal Loss (α=0.75, γ=2)** | 98.15% | **63.22%** | 96.99% | **99.32%** | **76.55%** |
| **B3** | BCE Baseline | 96.71% | 48.81% | **97.40%** | 96.01% | 65.03% |
| **B3** | **Focal Loss (α=0.75, γ=2)** | **97.10%** | **59.38%** | 96.79% | **97.42%** | **73.61%** |
| **B4** | BCE Baseline | 70.30% | **26.35%** | 41.24% | **99.36%** | **32.16%** |
| **B4** | **Focal Loss (α=0.75, γ=2)** | **96.31%** | 18.60% | **94.92%** | 97.70% | 31.11% |
| **B5** | BCE Baseline | 64.83% | 0.30% | **97.56%** | 32.10% | 0.61% |
| **B5** | **Focal Loss (α=0.75, γ=2)** | **86.24%** | **0.82%** | **97.56%** | **74.91%** | **1.62%** |

### 3. Ablation Study: Online Hard Example Mining (OHEM) on B4 & B5

| Benchmark | Loss & Sampling Strategy | Balanced Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **B4** | BCE Baseline | 70.30% | 26.35% | 41.24% | 32.16% |
| **B4** | **Focal Loss (Primary)** | **96.31%** | 18.60% | **94.92%** | 31.11% |
| **B4** | OHEM (Top 50% Loss) | 80.71% | **32.35%** | 62.15% | **42.55%** |
| **B4** | Focal + OHEM | 94.38% | 23.39% | 90.40% | 37.17% |
| **B5** | BCE Baseline | 64.83% | 0.30% | **97.56%** | 0.61% |
| **B5** | **Focal Loss (Primary)** | 86.24% | 0.82% | **97.56%** | 1.62% |
| **B5** | OHEM (Top 50% Loss) | 64.51% | **20.00%** | 29.27% | 23.76% |
| **B5** | **Focal + OHEM** | **90.34%** | 1.21% | **97.56%** | **2.39%** |

---

## 🏗️ Model Architecture & Mathematical Loss Formulations

### 1. Lightweight CNN Architecture (~12,873 Parameters)
```text
Input Clip (64x64x1 Grayscale)
  │
  ├── Block 1: Conv2D(3x3, 12) ──> ELU ──> Conv2D(3x3, 12) ──> ELU ──> Conv2D(3x3, 12) ──> BatchNorm ──> ELU ──> MaxPool(2x2)
  │
  ├── Block 2: Conv2D(3x3, 12) ──> ELU ──> Conv2D(3x3, 12) ──> ELU ──> Conv2D(3x3, 12) ──> BatchNorm ──> ELU ──> MaxPool(2x2)
  │
  ├── Dropout (p = 0.30)
  ├── Fully Connected Layer (10 units, ELU activation)
  └── FC Output Layer (1 unit, Sigmoid activation) ──> Probability p
```

### 2. Loss Formulations
* **Binary Cross-Entropy (BCE):**
  $$\mathcal{L}_{\text{BCE}} = - \big[ y \log(p) + (1-y) \log(1-p) \big]$$
* **Focal Loss ($\alpha=0.75, \gamma=2.0$):**
  $$\mathcal{L}_{\text{Focal}} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
  *where $p_t = p$ for positive hotspot samples, and $p_t = 1 - p$ for non-hotspot samples.*
* **Online Hard Example Mining (OHEM):**
  Computes per-sample losses across mini-batch size $N=128$, ranks magnitudes, and backpropagates gradients using **only the top 50% highest-loss samples**.

---

## 📁 Repository Directory Structure

```text
FINAL G10_DA2_COMPLETE_BACKUP/
├── checkpoints/                        # Model weights checkpoints (.pt)
│   ├── b1_bce.pt, b1_focal.pt
│   ├── b2_bce.pt, b2_focal.pt
│   ├── b3_bce.pt, b3_focal.pt
│   ├── b4_bce.pt, b4_focal.pt, b4_ohem.pt, b4_focal_ohem.pt
│   └── b5_bce.pt, b5_focal.pt, b5_ohem.pt, b5_focal_ohem.pt
├── plots/                              # Comparative graphics & confusion matrix heatmaps
│   ├── B1_Balanced_Accuracy.png ... B5_Balanced_Accuracy.png
│   ├── average_Balanced_Accuracy.png ... average_F1.png
│   └── confusion_matrices_b4_b5.png
├── train_g10.py                        # Main BCE & Focal Loss training script (B1–B5)
├── train_g10_ohem.py                   # OHEM & Focal+OHEM training script
├── evaluate_g10.py                     # Evaluation script for BCE & Focal Loss checkpoints
├── evaluate_g10_ohem.py                # Evaluation script for OHEM checkpoints
├── consolidate_results.py              # Metric consolidation script
├── generate_plots.py                   # Plotting script for accuracy/recall/precision charts
├── generate_confusion_matrices.py      # Confusion matrix heatmap generator
├── generate_analysis.py                # Statistical metric analysis generator
├── results_all_experiments.csv         # Tabular result CSV dump (raw counts & metrics)
├── G10_all_results.json                # JSON result file
├── requirements.txt                    # Python environment requirements
└── README.md                           # Documentation & execution instructions
```

---

## 🚀 Execution & Reproducibility Guide

### 1. Installation

Ensure Python 3.8+ and PyTorch are installed:

```bash
pip install -r requirements.txt
```

### 2. Dataset Layout Setup

Organize the official ICCAD-12 benchmarks as follows:

```text
iccad-official/
  ├── iccad1/
  │   ├── train/ (train_hs/*.png, train_nhs/*.png)
  │   └── test/  (test_hs/*.png, test_nhs/*.png)
  ├── iccad2/ ... iccad5/
```

### 3. Training Commands

To train Baseline BCE or Focal Loss on Benchmark 1 to 5:

```bash
# Baseline BCE Training (Benchmark 1)
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode bce --epochs 5

# Proposed Focal Loss Training (Benchmark 1)
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode focal --epochs 5
```

To run OHEM experiments on Benchmark 4 or 5:

```bash
python train_g10_ohem.py --root /path/to/iccad-official --benchmark 4 --mode ohem --epochs 5
python train_g10_ohem.py --root /path/to/iccad-official --benchmark 4 --mode focal_ohem --epochs 5
```

### 4. Evaluation Commands

Evaluate a saved checkpoint on the official untouched test set:

```bash
python evaluate_g10.py --root /path/to/iccad-official --benchmark 1 --checkpoint checkpoints/b1_focal.pt
```

To regenerate all comparative plots and confusion matrix graphics:

```bash
python generate_plots.py
python generate_confusion_matrices.py
```

---

## ⚠️ Reproducibility Caveat & Limitations

1. **OHEM Split Enumeration:** `train_g10_ohem.py` uses a separate file enumeration order compared to `train_g10.py`. While both scripts use the same random seed (42), stratification rules, $64\times 64$ grayscale preprocessing, CNN architecture, and Nadam optimizer, the individual train/validation image assignments differ slightly between OHEM and BCE/Focal runs.
2. **Fixed Sigmoid Threshold:** All predictions use a static $0.50$ decision threshold. Benchmark-specific threshold calibration on PR validation curves remains a recommendation for future research.
3. **Training Budget:** Model budget was fixed at 5 epochs as specified by the baseline assignment setup.

---

## 📚 References & Literature Survey

1. V. Borisov and J. Scheible, *"Lithography Hotspots Detection Using Deep Learning,"* Proc. 15th SMACD, 2018, pp. 145–148.
2. H. Yang, Y. Lin, B. Yu, and E. F. Y. Young, *"Lithography hotspot detection: From shallow to deep learning,"* Proc. IEEE SOCC, 2017, pp. 233–238.
3. L. Liao, S. Li, Y. Che, W. Shi, and X. Wang, *"Lithography Hotspot Detection Method Based on Transfer Learning Using Pre-Trained Deep Convolutional Neural Network,"* Applied Sciences, vol. 12, no. 4, 2192, 2022.
4. Y. Chen et al., *"Lightweight Hotspot Detection Model Fusing SE and ECA Mechanisms,"* Micromachines, vol. 15, no. 10, 1217, 2024.
5. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, *"Focal Loss for Dense Object Detection,"* Proc. IEEE ICCV, 2017, pp. 2980–2988.
6. A. Shrivastava, A. Gupta, and R. Girshick, *"Training Region-Based Object Detectors With Online Hard Example Mining,"* Proc. IEEE CVPR, 2016, pp. 761–769.

---

## 📜 License

This codebase and reproducibility suite are released under the MIT License for academic research purposes.
