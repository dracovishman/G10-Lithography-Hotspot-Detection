
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs("plots/confusion_matrices", exist_ok=True)

df = pd.read_csv("results_all_experiments.csv")

method_names = {
    "bce": "BCE",
    "focal": "Focal Loss",
    "ohem": "OHEM",
    "focal_ohem": "Focal + OHEM"
}

for _, row in df.iterrows():

    cm = np.array([
        [row["TN"], row["FP"]],
        [row["FN"], row["TP"]]
    ])

    plt.figure(figsize=(5,5))

    plt.imshow(cm)

    plt.title(
        f"B{int(row['Benchmark'])} — "
        f"{method_names.get(row['Method'], row['Method'])}"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        [0,1],
        ["Non-Hotspot", "Hotspot"]
    )

    plt.yticks(
        [0,1],
        ["Non-Hotspot", "Hotspot"]
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{int(cm[i,j]):,}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    filename = (
        f"B{int(row['Benchmark'])}_"
        f"{row['Method']}_confusion_matrix.png"
    )

    plt.savefig(
        f"plots/confusion_matrices/{filename}",
        dpi=300
    )

    plt.close()

print("Confusion matrices generated.")
