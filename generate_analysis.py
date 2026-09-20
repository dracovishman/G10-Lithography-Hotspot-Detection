
import pandas as pd
import numpy as np

df = pd.read_csv("results_all_experiments.csv")

metrics = [
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1"
]

method_names = {
    "bce": "BCE",
    "focal": "Focal Loss",
    "ohem": "OHEM",
    "focal_ohem": "Focal + OHEM"
}

lines = []

lines.append("G10 LITHOGRAPHY HOTSPOT DETECTION — EXPERIMENT ANALYSIS")
lines.append("=" * 70)
lines.append("")

# ------------------------------------------------------------
# Per benchmark
# ------------------------------------------------------------

for benchmark in sorted(df["Benchmark"].unique()):

    lines.append(f"BENCHMARK {int(benchmark)}")
    lines.append("-" * 70)

    b = df[df["Benchmark"] == benchmark]

    for _, row in b.iterrows():

        method = method_names.get(
            row["Method"],
            row["Method"]
        )

        lines.append(
            f"{method}: "
            f"BA={row['Balanced Accuracy']*100:.2f}%, "
            f"Precision={row['Precision']*100:.2f}%, "
            f"Recall={row['Recall']*100:.2f}%, "
            f"Specificity={row['Specificity']*100:.2f}%, "
            f"F1={row['F1']*100:.2f}%, "
            f"TN={int(row['TN'])}, "
            f"FP={int(row['FP'])}, "
            f"FN={int(row['FN'])}, "
            f"TP={int(row['TP'])}"
        )

    lines.append("")

# ------------------------------------------------------------
# Average
# ------------------------------------------------------------

avg = df.groupby("Method")[metrics].mean()

lines.append("FIVE-BENCHMARK AVERAGES")
lines.append("-" * 70)

for method in avg.index:

    lines.append(
        f"{method_names.get(method, method)}: "
        + ", ".join(
            f"{m}={avg.loc[method,m]*100:.2f}%"
            for m in metrics
        )
    )

lines.append("")

# ------------------------------------------------------------
# Focal vs BCE
# ------------------------------------------------------------

if "bce" in avg.index and "focal" in avg.index:

    lines.append("FOCAL LOSS VS BCE")
    lines.append("-" * 70)

    for metric in metrics:

        difference = (
            avg.loc["focal", metric]
            - avg.loc["bce", metric]
        ) * 100

        lines.append(
            f"{metric}: "
            f"{difference:+.2f} percentage points"
        )

    lines.append("")

# ------------------------------------------------------------
# B4
# ------------------------------------------------------------

b4 = df[df["Benchmark"] == 4].set_index("Method")

if "bce" in b4.index and "focal" in b4.index:

    lines.append("B4 BCE → FOCAL")
    lines.append("-" * 70)

    for metric in metrics:

        diff = (
            b4.loc["focal", metric]
            - b4.loc["bce", metric]
        ) * 100

        lines.append(
            f"{metric}: {diff:+.2f} pp"
        )

    lines.append("")

# ------------------------------------------------------------
# B5
# ------------------------------------------------------------

b5 = df[df["Benchmark"] == 5].set_index("Method")

if "bce" in b5.index and "focal" in b5.index:

    lines.append("B5 BCE → FOCAL")
    lines.append("-" * 70)

    for metric in metrics:

        diff = (
            b5.loc["focal", metric]
            - b5.loc["bce", metric]
        ) * 100

        lines.append(
            f"{metric}: {diff:+.2f} pp"
        )

    lines.append("")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

with open(
    "experiment_analysis.txt",
    "w"
) as f:

    f.write("\n".join(lines))

print("\n".join(lines))

print("\nSaved experiment_analysis.txt")
