# G10 Experimental Plan

## Controlled variables

- Dataset: ICCAD-12, benchmark-wise
- Input: grayscale, resized to 64x64
- Architecture: same lightweight CNN
- Optimizer: Nadam
- Learning rate: 1e-3
- Dropout: 0.3
- Random seed: 42
- Validation split: 20% stratified from training data
- Test set: official test set only
- Threshold: sigmoid >= 0.5
- Primary metric: Balanced Accuracy

## Main comparison

BCE vs Focal vs OHEM vs Focal+OHEM.

## Focal-loss ablation

Run gamma = 0, 1, 2, 3 while holding alpha and all other conditions fixed.

## OHEM ablation

Run hard_fraction = 1.0, 0.75, 0.50, 0.25.

## Difficult-sample analysis

Store prediction probabilities and classify samples into:

- easy correct: confidence >= 0.90
- moderate correct: 0.60 <= confidence < 0.90
- hard correct: confidence < 0.60
- misclassified

The report should compare hotspot recall and false negatives for these categories.

## Reporting

For every benchmark and every main model, report:

TN, FP, FN, TP, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity and F1.

Do not replace actual experimental results with results from published papers.
