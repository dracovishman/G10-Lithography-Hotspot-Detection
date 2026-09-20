
import json
import glob
import os
import pandas as pd

records = []

for file in sorted(glob.glob("results_b*.json")):

    with open(file, "r") as f:
        d = json.load(f)

    benchmark = d.get("benchmark")

    mode = d.get("mode")

    if mode is None:
        filename = os.path.basename(file)

        if "focal_ohem" in filename:
            mode = "focal_ohem"
        elif "ohem" in filename:
            mode = "ohem"
        elif "focal" in filename:
            mode = "focal"
        elif "bce" in filename:
            mode = "bce"

    records.append({
        "Benchmark": benchmark,
        "Method": mode,
        "TN": d.get("TN"),
        "FP": d.get("FP"),
        "FN": d.get("FN"),
        "TP": d.get("TP"),
        "Balanced Accuracy": d.get("balanced_accuracy"),
        "Precision": d.get("precision"),
        "Recall": d.get("recall"),
        "Specificity": d.get("specificity"),
        "F1": d.get("f1"),
        "Test Samples": d.get("test_samples")
    })

df = pd.DataFrame(records)

method_order = [
    "bce",
    "focal",
    "ohem",
    "focal_ohem"
]

df["Method"] = pd.Categorical(
    df["Method"],
    categories=method_order,
    ordered=True
)

df = df.sort_values(["Benchmark", "Method"])

# Save raw consolidated table
df.to_csv(
    "results_all_experiments.csv",
    index=False
)

# Percentage version
metrics = [
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1"
]

df_percent = df.copy()

for col in metrics:
    df_percent[col] = df_percent[col] * 100

df_percent.to_csv(
    "results_all_experiments_percent.csv",
    index=False
)

print("\n" + "=" * 80)
print("ALL EXPERIMENT RESULTS")
print("=" * 80)

print(df_percent.to_string(index=False))

# ------------------------------------------------------------
# Average across five benchmarks
# ------------------------------------------------------------

average = (
    df.groupby("Method", observed=True)[metrics]
    .mean()
    .reset_index()
)

average_percent = average.copy()

for col in metrics:
    average_percent[col] *= 100

average_percent.to_csv(
    "results_average_by_method.csv",
    index=False
)

print("\n" + "=" * 80)
print("FIVE-BENCHMARK AVERAGES")
print("=" * 80)

print(average_percent.to_string(index=False))

# ------------------------------------------------------------
# BCE vs Focal
# ------------------------------------------------------------

pivot = average.set_index("Method")

if "bce" in pivot.index and "focal" in pivot.index:

    delta = (
        pivot.loc["focal"] -
        pivot.loc["bce"]
    )

    print("\n" + "=" * 80)
    print("FOCAL - BCE IMPROVEMENT")
    print("=" * 80)

    for metric in metrics:
        print(
            f"{metric}: "
            f"{delta[metric]*100:+.2f} percentage points"
        )

# ------------------------------------------------------------
# B4 detailed comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("B4 COMPARISON")
print("=" * 80)

print(
    df_percent[
        df_percent["Benchmark"] == 4
    ].to_string(index=False)
)

# ------------------------------------------------------------
# B5 detailed comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("B5 COMPARISON")
print("=" * 80)

print(
    df_percent[
        df_percent["Benchmark"] == 5
    ].to_string(index=False)
)

print("\nSaved:")
print("  results_all_experiments.csv")
print("  results_all_experiments_percent.csv")
print("  results_average_by_method.csv")
