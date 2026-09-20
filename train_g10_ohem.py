
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

        image = np.asarray(image, dtype=np.float32) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


def collect_dataset(root, benchmark):

    benchmark_dir = os.path.join(
        root,
        f"iccad{benchmark}"
    )

    hs_dir = os.path.join(
        benchmark_dir,
        "train",
        "train_hs"
    )

    nhs_dir = os.path.join(
        benchmark_dir,
        "train",
        "train_nhs"
    )

    paths = []
    labels = []

    for filename in sorted(os.listdir(nhs_dir)):

        path = os.path.join(
            nhs_dir,
            filename
        )

        paths.append(path)
        labels.append(0)

    for filename in sorted(os.listdir(hs_dir)):

        path = os.path.join(
            hs_dir,
            filename
        )

        paths.append(path)
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

            nn.Linear(12 * 11 * 11, 10),

            nn.ELU(),

            nn.Linear(10, 1)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# Per-sample losses
# ============================================================

def bce_per_sample(logits, targets):

    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )


def focal_per_sample(
    logits,
    targets,
    alpha=0.75,
    gamma=2.0
):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    probabilities = torch.sigmoid(logits)

    pt = torch.where(
        targets == 1,
        probabilities,
        1.0 - probabilities
    )

    alpha_factor = torch.where(
        targets == 1,
        torch.full_like(targets, alpha),
        torch.full_like(targets, 1.0 - alpha)
    )

    loss = (
        alpha_factor
        * ((1.0 - pt) ** gamma)
        * bce
    )

    return loss


# ============================================================
# OHEM loss
# ============================================================

def ohem_loss(
    per_sample_loss,
    hard_fraction
):

    batch_size = per_sample_loss.numel()

    hard_fraction = max(
        min(hard_fraction, 1.0),
        0.05
    )

    k = max(
        1,
        int(np.ceil(batch_size * hard_fraction))
    )

    hard_losses, _ = torch.topk(
        per_sample_loss,
        k=k,
        largest=True
    )

    return hard_losses.mean()


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate_validation(
    model,
    loader,
    device
):

    model.eval()

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).long()

        all_labels.extend(
            labels.cpu().numpy().astype(int)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)

    tp = np.sum(
        (labels == 1) &
        (predictions == 1)
    )

    tn = np.sum(
        (labels == 0) &
        (predictions == 0)
    )

    fp = np.sum(
        (labels == 0) &
        (predictions == 1)
    )

    fn = np.sum(
        (labels == 1) &
        (predictions == 0)
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    return balanced_accuracy


# ============================================================
# Training
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
        "--mode",
        choices=[
            "ohem",
            "focal_ohem"
        ],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--hard_fraction",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
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
    # Dataset
    # --------------------------------------------------------

    paths, labels = collect_dataset(
        args.root,
        args.benchmark
    )

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.20,
            random_state=SEED,
            stratify=labels
        )
    )

    train_dataset = HotspotDataset(
        train_paths,
        train_labels
    )

    val_dataset = HotspotDataset(
        val_paths,
        val_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HotspotCNN().to(device)

    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=1e-3
    )

    best_val_ba = -1.0
    best_epoch = -1

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    checkpoint_path = (
        f"checkpoints/"
        f"b{args.benchmark}_{args.mode}.pt"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        model.train()

        running_loss = 0.0
        batches = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            if args.mode == "ohem":

                losses = bce_per_sample(
                    logits,
                    labels
                )

            else:

                losses = focal_per_sample(
                    logits,
                    labels,
                    alpha=args.alpha,
                    gamma=args.gamma
                )

            loss = ohem_loss(
                losses,
                args.hard_fraction
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batches += 1

        val_ba = evaluate_validation(
            model,
            val_loader,
            device
        )

        average_loss = (
            running_loss / batches
            if batches > 0
            else 0.0
        )

        print(
            f"benchmark={args.benchmark} "
            f"mode={args.mode} "
            f"epoch={epoch} "
            f"loss={average_loss:.6f} "
            f"val_BA={val_ba:.6f}"
        )

        if val_ba > best_val_ba:

            best_val_ba = val_ba
            best_epoch = epoch

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "benchmark":
                    args.benchmark,

                "mode":
                    args.mode,

                "best_epoch":
                    best_epoch,

                "val_balanced_accuracy":
                    best_val_ba,

                "hard_fraction":
                    args.hard_fraction,

                "gamma":
                    args.gamma,

                "alpha":
                    args.alpha
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

    print(
        f"saved: {checkpoint_path}"
    )

    print(
        f"best_epoch={best_epoch}"
    )

    print(
        f"best_val_balanced_accuracy="
        f"{best_val_ba:.6f}"
    )


if __name__ == "__main__":
    main()
