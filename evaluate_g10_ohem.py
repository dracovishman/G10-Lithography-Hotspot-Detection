
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class HotspotDataset(Dataset):

    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert("L")
        image = image.resize((64, 64))

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


def collect_test_dataset(root, benchmark):

    benchmark_dir = os.path.join(
        root,
        f"iccad{benchmark}"
    )

    hs_dir = os.path.join(
        benchmark_dir,
        "test",
        "test_hs"
    )

    nhs_dir = os.path.join(
        benchmark_dir,
        "test",
        "test_nhs"
    )

    paths = []
    labels = []

    for filename in sorted(os.listdir(nhs_dir)):

        paths.append(
            os.path.join(
                nhs_dir,
                filename
            )
        )

        labels.append(0)

    for filename in sorted(os.listdir(hs_dir)):

        paths.append(
            os.path.join(
                hs_dir,
                filename
            )
        )

        labels.append(1)

    return paths, labels


# ============================================================
# Model
# ============================================================

class BasicBlock(nn.Module):

    def __init__(self, in_channels, channels=12):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3
            ),

            nn.ELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3
            ),

            nn.ELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3
            ),

            nn.BatchNorm2d(channels),

            nn.ELU(),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class HotspotCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            BasicBlock(1, 12),
            BasicBlock(12, 12)
        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Flatten(),

            nn.Linear(
                12 * 11 * 11,
                10
            ),

            nn.ELU(),

            nn.Linear(10, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# Evaluation
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True
    )

    parser.add_argument(
        "--benchmark",
        type=int,
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    print(
        "Checkpoint mode:",
        checkpoint.get("mode", "unknown")
    )

    print(
        "Best epoch:",
        checkpoint.get("best_epoch", "unknown")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HotspotCNN().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    paths, labels = collect_test_dataset(
        args.root,
        args.benchmark
    )

    dataset = HotspotDataset(
        paths,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    start = time.time()

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                batch_labels.numpy().astype(int)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    inference_seconds = time.time() - start

    labels = np.asarray(
        all_labels
    )

    predictions = np.asarray(
        all_predictions
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = int(np.sum(
        (labels == 1) &
        (predictions == 1)
    ))

    tn = int(np.sum(
        (labels == 0) &
        (predictions == 0)
    ))

    fp = int(np.sum(
        (labels == 0) &
        (predictions == 1)
    ))

    fn = int(np.sum(
        (labels == 1) &
        (predictions == 0)
    ))

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,

        "balanced_accuracy":
            balanced_accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "benchmark":
            args.benchmark,

        "mode":
            checkpoint.get(
                "mode",
                "ohem"
            ),

        "best_epoch":
            checkpoint.get(
                "best_epoch"
            ),

        "val_balanced_accuracy":
            checkpoint.get(
                "val_balanced_accuracy"
            ),

        "hard_fraction":
            checkpoint.get(
                "hard_fraction"
            ),

        "test_samples":
            len(labels),

        "inference_seconds":
            inference_seconds,

        "inference_ms_per_sample":
            1000.0 *
            inference_seconds /
            len(labels)
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        f"results_b{args.benchmark}_"
        f"{checkpoint.get('mode', 'ohem')}.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"TP: {tp}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.6f} "
        f"({balanced_accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: "
        f"{precision:.6f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall: "
        f"{recall:.6f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Specificity: "
        f"{specificity:.6f} "
        f"({specificity * 100:.2f}%)"
    )

    print(
        f"F1: "
        f"{f1:.6f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Test samples: {len(labels)}"
    )

    print(
        f"Inference time: "
        f"{inference_seconds:.2f} seconds"
    )

    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()
