# G10 — Focal and Hard-Example Learning for Lithography Hotspot Detection

Official implementation and experimental reproducibility benchmark for **Group 10 (G10)** — Digital Assignment II (DA-2), Course: *AI and Machine Learning for IC Design* (BEVD402L), Fall 2026-27, Vellore Institute of Technology (VIT Chennai).

---

## 📌 Project Overview

Lithography hotspot detection identifies layout patterns that are susceptible to manufacturing defects (open circuit pinching or short circuit bridging) during photolithographic processing. Due to severe class imbalance in ICCAD-12 benchmarks ($<1\%$ hotspots in test sets), conventional Binary Cross-Entropy (BCE) loss training suffers from easy-negative gradient dominance.

This project investigates training-level hard-example emphasis using **Focal Loss** ($\alpha=0.75, \gamma=2.0$) and **Online Hard Example Mining (OHEM)** while maintaining a fixed **Lightweight CNN architecture (~12,873 parameters)** for fast CAD tool inference.

---

## 👥 Group 10 Team Members

| Reg. No. | Student Name | Role / Key Contribution |
| :--- | :--- | :--- |
| **23BVD1062** | Vinayak Shreenivas Salunke | Dataset Preprocessing, BCE & Focal Loss Baseline Execution, Result Aggregation |
| **23BVD1064** | Harsh Vardhan Singh | OHEM Architecture Formulation, IEEE Report Writing, Discussion & Limitation Analysis |
| **23BVD1065** | Alavala Vishnu Koushik Reddy | Confusion Matrix Heatmaps, Comparative Plot Generation, Presentation Deck Preparation |

* **Faculty Supervisor:** Dr. G. Lakshmi Priya
* **Course:** AI & Machine Learning for IC Design (BEVD402L) | Slot: D1+TD1

---

## 📊 Summary of Key Experimental Results

Across all five ICCAD-12 benchmarks (B1–B5), proposed **Focal Loss** achieves a **+10.24 percentage point increase in arithmetic-mean Balanced Accuracy** over the standard BCE baseline without introducing any parameter or architectural overhead:

| Metric | BCE Baseline (Mean) | Proposed Focal Loss (Mean) | Improvement |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy (Primary)** | **82.78%** | **93.02%** | **+10.24%** |
| **Recall / Sensitivity** | **86.73%** | **96.45%** | **+9.72%** |
| **Specificity** | **78.84%** | **89.58%** | **+10.74%** |
| **Precision** | **27.07%** | **31.96%** | **+4.89%** |
| **F1-Score** | **36.88%** | **42.58%** | **+5.70%** |

---

## 📁 Repository Directory Structure

```text
FINAL G10_DA2_COMPLETE_BACKUP/
├── checkpoints/                        # Saved model weight checkpoints (.pt)
│   ├── b1_bce.pt, b1_focal.pt
│   ├── b2_bce.pt, b2_focal.pt
│   ├── b3_bce.pt, b3_focal.pt
│   ├── b4_bce.pt, b4_focal.pt, b4_ohem.pt, b4_focal_ohem.pt
│   └── b5_bce.pt, b5_focal.pt, b5_ohem.pt, b5_focal_ohem.pt
├── plots/                              # Comparative bar charts and confusion matrix heatmaps
├── train_g10.py                        # Main training pipeline (BCE & Focal Loss for B1-B5)
├── train_g10_ohem.py                   # OHEM and Focal+OHEM training pipeline
├── evaluate_g10.py                     # Evaluation script for test set metrics (BCE & Focal)
├── evaluate_g10_ohem.py                # Evaluation script for OHEM variants
├── consolidate_results.py              # Result aggregation & CSV consolidation
├── generate_plots.py                   # Plotting script for accuracy/recall/precision graphics
├── generate_confusion_matrices.py      # Confusion matrix generation script
├── generate_analysis.py                # Quantitative statistical analysis generator
├── results_all_experiments.csv         # Consolidated tabular results (raw counts & metrics)
├── G10_all_results.json                # JSON dump of full experimental metrics
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Documentation & execution instructions
```

---

## 🚀 How to Run & Reproduce Experiments

### 1. Prerequisites & Installation

Ensure Python 3.8+ and PyTorch are installed:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Organize the official ICCAD-12 benchmarks in the following structure:

```text
iccad-official/
  ├── iccad1/
  │   ├── train/ (train_hs/*.png, train_nhs/*.png)
  │   └── test/  (test_hs/*.png, test_nhs/*.png)
  ├── iccad2/ ... iccad5/
```

### 3. Training Models

To train Baseline BCE or Focal Loss on Benchmark 1:

```bash
# Train BCE Baseline
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode bce --epochs 5

# Train Focal Loss
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode focal --epochs 5
```

To train OHEM variants on Benchmark 4 or 5:

```bash
python train_g10_ohem.py --root /path/to/iccad-official --benchmark 4 --mode ohem --epochs 5
python train_g10_ohem.py --root /path/to/iccad-official --benchmark 4 --mode focal_ohem --epochs 5
```

### 4. Evaluation & Metric Generation

Evaluate a trained model checkpoint on the test set:

```bash
python evaluate_g10.py --root /path/to/iccad-official --benchmark 1 --checkpoint checkpoints/b1_focal.pt
```

To regenerate all comparative plots and confusion matrices:

```bash
python generate_plots.py
python generate_confusion_matrices.py
```

---

## 📚 References & Citation

1. V. Borisov and J. Scheible, *"Lithography Hotspots Detection Using Deep Learning,"* Proc. 15th SMACD, 2018, pp. 145–148.
2. T.-Y. Lin et al., *"Focal Loss for Dense Object Detection,"* Proc. IEEE ICCV, 2017, pp. 2980–2988.
3. A. Shrivastava et al., *"Training Region-Based Object Detectors With Online Hard Example Mining,"* Proc. IEEE CVPR, 2016, pp. 761–769.

---

## 📜 License
This repository is released under the MIT License for academic research purposes.
