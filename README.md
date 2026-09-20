# G10 — Focal and Hard-Example Learning for Lithography Hotspot Detection on ICCAD-12

[![Course](https://img.shields.io/badge/Course-BEVD402L--AI%20%26%20ML%20for%20IC-blue.svg)](https://vit.ac.in)
[![Institution](https://img.shields.io/badge/Institution-VIT%20Chennai-orange.svg)](https://vit.ac.in)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

Official source code, experimental reproducibility suite, and benchmark evaluation for **Group 10 (G10)** — Digital Assignment II (DA-2):
* **Course:** AI and Machine Learning for IC Design (BEVD402L) | Slot: D1+TD1
* **Faculty Supervisor:** Dr. G. Lakshmi Priya
* **Institution:** School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT), Chennai

---

## 📌 1. Project Overview & Common Research Context

Lithography hotspot detection identifies layout topology clips susceptible to manufacturing printability defects (such as open circuit line-end pinching or short circuit line bridging) during photolithographic wafer processing. As semiconductor technology nodes shrink below 20nm, the mismatch between lithography laser wavelength (193nm) and nanometer feature sizes causes optical diffraction effects, degrading chip yield.

### The ICCAD-12 Benchmark Suite
Introduced in the **2012 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)** CAD Contest, the ICCAD-12 suite contains five distinct benchmark layout datasets (B1 to B5). The dataset is characterized by severe class imbalance where hotspot patterns (HS) are heavily outnumbered by non-hotspot patterns (NHS) (e.g., Benchmark 5 test set contains only 41 hotspots out of 19,368 total clips, representing a $0.21\%$ hotspot prevalence).

### Standard Accuracy Fallacy
Under such extreme class imbalance, standard accuracy is misleading: a trivial classifier predicting 100% Non-Hotspot scores over **99% standard accuracy** while achieving **0% Hotspot Recall**. Therefore, **Balanced Accuracy (BA)** is mandated as the primary quality metric:
$$\text{Balanced Accuracy (BA)} = \frac{\text{Sensitivity (Recall)} + \text{Specificity}}{2}$$

---

## 👥 2. Group 10 Team Members & Work Allocation

| Reg. No. | Student Name | Role / Key Module Contribution | Contribution % |
| :---: | :--- | :--- | :---: |
| **23BVD1062** | Vinayak Shreenivas Salunke | Dataset Preprocessing, BCE & Focal Loss Baseline Execution, Result Aggregation | **33.3%** |
| **23BVD1064** | Harsh Vardhan Singh | OHEM Architecture Formulation, IEEE Report Writing, Discussion & Limitation Analysis | **33.3%** |
| **23BVD1065** | Alavala Vishnu Koushik Reddy | Confusion Matrix Heatmaps, Comparative Plot Generation, Presentation Deck Preparation | **33.3%** |

---

## 🎯 3. Group 10 Research Questions, Hypotheses & Mandate Verification

### Research Questions
* **RQ1:** Does Focal Loss improve hotspot detection reliability compared with standard Binary Cross-Entropy (BCE) training across all five ICCAD-12 benchmarks?
* **RQ2:** Does explicit Online Hard Example Mining (OHEM) provide additive benefit over Focal Loss alone or when combined with Focal Loss?

### Hypotheses
* **H1:** Focal Loss will improve balanced accuracy and minority-class recall relative to BCE because easy non-hotspot samples contribute negligible loss gradients during backpropagation.
* **H2:** Combining Focal Loss with OHEM may provide complementary hard-example emphasis, though the effect depends on benchmark difficulty and precision-recall operating points.

### Faculty Mandate Verification Summary
* **✔ Research Question Answered:** YES. Focal Loss boosts 5-benchmark mean Balanced Accuracy from **82.78% to 93.02%** (+10.24 percentage points) and mean Hotspot Recall from **86.73% to 96.45%**.
* **✔ Research Focus & Novelty Fulfilled:** Investigated training-level loss weighting (Focal Loss $\alpha=0.75, \gamma=2.0$) and mini-batch mining (OHEM top 50%) without changing network capacity or adding parameter overhead.
* **✔ Minimum Expected Investigation Fulfilled:** Conducted a controlled BCE vs. Focal Loss comparison across all five benchmarks (B1–B5) and detailed minority sample error analysis (B4 missed hotspots reduced from 104 to 9; B5 false alarms reduced from 13,123 to 4,849).

---

## 📊 4. Summary Performance Comparison Across ICCAD-12

### A. Overall Five-Benchmark Arithmetic Mean Metrics

| Primary Evaluation Metric | BCE Baseline (Mean) | Proposed Focal Loss (Mean) | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy (Primary)** | **82.78%** | **93.02%** | **+10.24 pp** |
| **Recall / Sensitivity** | **86.73%** | **96.45%** | **+9.72 pp** |
| **Specificity** | **78.84%** | **89.58%** | **+10.74 pp** |
| **Precision** | **27.07%** | **31.96%** | **+4.89 pp** |
| **F1-Score** | **36.88%** | **42.58%** | **+5.70 pp** |

### B. Benchmark-Wise Test Performance (B1 to B5)

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

### C. Ablation Study: Online Hard Example Mining (OHEM) on B4 & B5

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

## 🏗️ 5. Model Architecture & Mathematical Loss Formulations

### Lightweight CNN Architecture (~12,873 Parameters)
```text
Input Clip Image (64x64x1 Grayscale)
  │
  ├── Basic Block 1:
  │     Conv2D (3x3 kernel, 12 channels, ELU activation)
  │     Conv2D (3x3 kernel, 12 channels, ELU activation)
  │     Conv2D (3x3 kernel, 12 channels) ──> BatchNorm ──> ELU ──> MaxPool2D(2x2)
  │
  ├── Basic Block 2:
  │     Conv2D (3x3 kernel, 12 channels, ELU activation)
  │     Conv2D (3x3 kernel, 12 channels, ELU activation)
  │     Conv2D (3x3 kernel, 12 channels) ──> BatchNorm ──> ELU ──> MaxPool2D(2x2)
  │
  ├── Flatten ──> Dropout (rate = 0.30)
  ├── Fully Connected Layer (10 units, ELU activation)
  └── FC Output Layer (1 unit, Sigmoid activation) ──> Predicted Probability p
```

### Mathematical Formulations
* **Binary Cross-Entropy (BCE) Baseline Loss:**
  $$\mathcal{L}_{\text{BCE}} = - \big[ y \log(p) + (1-y) \log(1-p) \big]$$
* **Focal Loss ($\alpha=0.75, \gamma=2.0$):**
  $$\mathcal{L}_{\text{Focal}} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
  *where $p_t = p$ for positive hotspot samples, and $p_t = 1 - p$ for non-hotspot samples.*
* **Online Hard Example Mining (OHEM):**
  Computes per-sample losses across batch size $N=128$, ranks loss magnitudes, and backpropagates gradients using **only the top 50% highest-loss samples**.

---

## 📁 6. Repository Directory Structure & Checkpoint Map

```text
FINAL G10_DA2_COMPLETE_BACKUP/
├── checkpoints/                        # Model weights checkpoints (.pt)
│   ├── b1_bce.pt, b1_focal.pt          # Benchmark 1 trained weights
│   ├── b2_bce.pt, b2_focal.pt          # Benchmark 2 trained weights
│   ├── b3_bce.pt, b3_focal.pt          # Benchmark 3 trained weights
│   ├── b4_bce.pt, b4_focal.pt          # Benchmark 4 trained weights
│   ├── b4_ohem.pt, b4_focal_ohem.pt    # Benchmark 4 OHEM ablation weights
│   ├── b5_bce.pt, b5_focal.pt          # Benchmark 5 trained weights
│   └── b5_ohem.pt, b5_focal_ohem.pt    # Benchmark 5 OHEM ablation weights
├── plots/                              # Visualization graphics & confusion matrix heatmaps
│   ├── B1_Balanced_Accuracy.png ... B5_Balanced_Accuracy.png
│   ├── average_Balanced_Accuracy.png ... average_F1.png
│   └── confusion_matrices_b4_b5.png
├── train_g10.py                        # Main BCE & Focal Loss training script (B1–B5)
├── train_g10_ohem.py                   # OHEM & Focal+OHEM training script
├── evaluate_g10.py                     # Test set evaluation script for BCE & Focal Loss
├── evaluate_g10_ohem.py                # Test set evaluation script for OHEM variants
├── consolidate_results.py              # Metric aggregation & CSV/JSON consolidation script
├── generate_plots.py                   # Comparative metric bar chart generator
├── generate_confusion_matrices.py      # Confusion matrix heatmap generator
├── generate_analysis.py                # Statistical metric analysis generator
├── results_all_experiments.csv         # Consolidated raw counts & evaluation metrics CSV
├── G10_all_results.json                # Structured JSON result file
├── requirements.txt                    # Python environment requirements
└── README.md                           # Documentation & execution instructions
```

---

## 🚀 7. Execution & Reproducibility Guide

### Step 1: Environment Setup & Prerequisites
Ensure Python 3.8+ and PyTorch are installed. A GPU is recommended for faster execution but CPU inference is fully supported:

```bash
pip install -r requirements.txt
```

### Step 2: Dataset Directory Setup
Organize the official ICCAD-12 benchmarks as follows (do NOT merge the benchmark folders):

```text
iccad-official/
  ├── iccad1/
  │   ├── train/ (train_hs/*.png, train_nhs/*.png)
  │   └── test/  (test_hs/*.png, test_nhs/*.png)
  ├── iccad2/ ... iccad5/
```
*Note: The ICCAD-12 dataset is not redistributed in this repository. Obtain the benchmark through course-provided resources.*

### Step 3: Training Commands
To train Baseline BCE or Proposed Focal Loss on Benchmarks 1 through 5:

```bash
# Train Baseline BCE (Benchmark 1)
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode bce --epochs 5

# Train Proposed Focal Loss (Benchmark 1)
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode focal --epochs 5
```

To run OHEM ablation experiments on Benchmark 4 or 5:

```bash
python train_g10_ohem.py --root /path/to/iccad-official --benchmark 4 --mode ohem --epochs 5
python train_g10_ohem.py --root /path/to/iccad-official --benchmark 4 --mode focal_ohem --epochs 5
```

### Step 4: Evaluation Commands
To evaluate a trained checkpoint on the official untouched test set:

```bash
python evaluate_g10.py --root /path/to/iccad-official --benchmark 1 --checkpoint checkpoints/b1_focal.pt
```

To regenerate all comparative bar charts and confusion matrix heatmaps:

```bash
python generate_plots.py
python generate_confusion_matrices.py
```

---

## ⚠️ 8. Reproducibility Caveats & Technical Limitations

1. **OHEM File Enumeration Order:** `train_g10_ohem.py` uses a separate file-enumeration order compared to `train_g10.py`. While both scripts maintain identical random seed (42), stratification rules, $64\times 64$ grayscale preprocessing, CNN architecture, and Nadam optimizer, the individual train/validation file splits differ slightly between OHEM and BCE/Focal runs.
2. **Fixed Decision Threshold:** Predictions use a fixed sigmoid threshold of $0.50$. Benchmark-specific threshold calibration on validation PR-curves remains a recommendation for future research.
3. **Hyperparameter Sweeps:** The primary Focal Loss experiment uses fixed $\alpha=0.75, \gamma=2.0$, and OHEM hard fraction $0.50$. Systematic parameter sweeps were beyond the 5-epoch fixed baseline budget.

---

## 📚 9. Literature References & Academic Citations

1. V. Borisov and J. Scheible, *"Lithography Hotspots Detection Using Deep Learning,"* Proc. 15th SMACD, 2018, pp. 145–148.
2. H. Yang, Y. Lin, B. Yu, and E. F. Y. Young, *"Lithography hotspot detection: From shallow to deep learning,"* Proc. IEEE SOCC, 2017, pp. 233–238.
3. L. Liao, S. Li, Y. Che, W. Shi, and X. Wang, *"Lithography Hotspot Detection Method Based on Transfer Learning Using Pre-Trained Deep Convolutional Neural Network,"* Applied Sciences, vol. 12, no. 4, 2192, 2022.
4. Y. Chen et al., *"Lightweight Hotspot Detection Model Fusing SE and ECA Mechanisms,"* Micromachines, vol. 15, no. 10, 1217, 2024.
5. M. Lin et al., *"An Improved YOLOv5 Model for Lithographic Hotspot Detection,"* Micromachines, vol. 16, no. 5, 568, 2025.
6. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, *"Focal Loss for Dense Object Detection,"* Proc. IEEE ICCV, 2017, pp. 2980–2988.
7. A. Shrivastava, A. Gupta, and R. Girshick, *"Training Region-Based Object Detectors With Online Hard Example Mining,"* Proc. IEEE CVPR, 2016, pp. 761–769.

---

## 📜 10. Academic Integrity & License
This repository is released under the **MIT License**. All experimental code, preprocessing pipelines, and evaluation metrics were developed independently by Group 10 in accordance with course academic integrity guidelines.
