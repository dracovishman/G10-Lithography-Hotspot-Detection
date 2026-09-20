
import os
import sys
import subprocess
import json
import time
from pathlib import Path

ROOT = "/content/dataset/iccad-official"
PROJECT = "/content/G10_project"

os.chdir(PROJECT)

def run(cmd, description):
    print("\n")
    print("=" * 80)
    print(description)
    print("=" * 80)
    print("COMMAND:")
    print(cmd)
    print()

    start = time.time()

    result = subprocess.run(
        cmd,
        shell=True,
        text=True
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n!!! FAILED: {description}")
        print(f"Return code: {result.returncode}")
        print("Continuing to next experiment...")
    else:
        print(f"\nCompleted: {description}")
        print(f"Time: {elapsed/60:.2f} minutes")

    return result.returncode


# ============================================================
# EXPERIMENT 1: B5 OHEM
# ============================================================

run(
    f"""python train_g10_ohem.py \
    --root {ROOT} \
    --benchmark 5 \
    --mode ohem \
    --epochs 5 \
    --batch_size 128 \
    --hard_fraction 0.50 \
    --gamma 2.0 \
    --alpha 0.75""",
    "B5 OHEM TRAINING"
)

if os.path.exists("checkpoints/b5_ohem.pt"):
    run(
        f"""python evaluate_g10_ohem.py \
        --root {ROOT} \
        --benchmark 5 \
        --checkpoint checkpoints/b5_ohem.pt""",
        "B5 OHEM EVALUATION"
    )


# ============================================================
# EXPERIMENT 2: B5 FOCAL + OHEM
# ============================================================

run(
    f"""python train_g10_ohem.py \
    --root {ROOT} \
    --benchmark 5 \
    --mode focal_ohem \
    --epochs 5 \
    --batch_size 128 \
    --hard_fraction 0.50 \
    --gamma 2.0 \
    --alpha 0.75""",
    "B5 FOCAL + OHEM TRAINING"
)

if os.path.exists("checkpoints/b5_focal_ohem.pt"):
    run(
        f"""python evaluate_g10_ohem.py \
        --root {ROOT} \
        --benchmark 5 \
        --checkpoint checkpoints/b5_focal_ohem.pt""",
        "B5 FOCAL + OHEM EVALUATION"
    )


print("\n")
print("=" * 80)
print("ALL REMAINING CORE EXPERIMENTS COMPLETED")
print("=" * 80)

print("\nExisting result files:")
for f in sorted(Path(".").glob("results_b*.json")):
    print("  ", f)

print("\nExisting checkpoints:")
for f in sorted(Path("checkpoints").glob("*.pt")):
    print("  ", f)
