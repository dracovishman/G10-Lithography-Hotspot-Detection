
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

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

# ============================================================
# 1. FIVE-BENCHMARK BALANCED ACCURACY
# ============================================================

for method in df["Method"].unique():

    subset = df[df["Method"] == method]

    plt.figure(figsize=(7,5))

    plt.plot(
        subset["Benchmark"],
        subset["Balanced Accuracy"] * 100,
        marker="o"
    )

    plt.xlabel("Benchmark")
    plt.ylabel("Balanced Accuracy (%)")
    plt.title(
        f"Balanced Accuracy Across ICCAD-12 Benchmarks: "
        f"{method_names.get(method, method)}"
    )

    plt.xticks([1,2,3,4,5])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"plots/BA_{method}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 2. METHOD COMPARISON — AVERAGE
# ============================================================

avg = (
    df.groupby("Method")[metrics]
    .mean()
)

for metric in metrics:

    plt.figure(figsize=(8,5))

    values = avg[metric] * 100

    plt.bar(
        [method_names.get(x,x) for x in values.index],
        values
    )

    plt.ylabel(f"{metric} (%)")
    plt.title(
        f"Average {metric} Across Five Benchmarks"
    )

    plt.ylim(0,100)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    safe_name = metric.replace(" ", "_")

    plt.savefig(
        f"plots/average_{safe_name}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 3. B4 METHOD COMPARISON
# ============================================================

b4 = df[df["Benchmark"] == 4].copy()

for metric in metrics:

    plt.figure(figsize=(8,5))

    values = b4.set_index("Method")[metric] * 100

    plt.bar(
        [method_names.get(x,x) for x in values.index],
        values
    )

    plt.ylabel(f"{metric} (%)")
    plt.title(f"B4: {metric} Comparison")

    plt.ylim(0,100)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"plots/B4_{metric.replace(' ','_')}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 4. B5 METHOD COMPARISON
# ============================================================

b5 = df[df["Benchmark"] == 5].copy()

for metric in metrics:

    plt.figure(figsize=(8,5))

    values = b5.set_index("Method")[metric] * 100

    plt.bar(
        [method_names.get(x,x) for x in values.index],
        values
    )

    plt.ylabel(f"{metric} (%)")
    plt.title(f"B5: {metric} Comparison")

    plt.ylim(0,100)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"plots/B5_{metric.replace(' ','_')}.png",
        dpi=300
    )

    plt.close()


print("Plots generated successfully.")

for f in sorted(os.listdir("plots")):
    print(" ", f)
