import subprocess
import random
import yaml
import json
import os
from pathlib import Path

# ===============================
# CONFIG
# ===============================
PROJECT_DIR = "yolo_tuning"
DATA_YAML = "data.yaml"
MODEL = "yolov9-c.pt"
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
DEVICE = 0

N_TRIALS = 20   # Like GridSearchCV but smarter

# ===============================
# SEARCH SPACE (Bayesian-inspired)
# ===============================
def sample_hyperparameters():
    return {
        "lr0": 10 ** random.uniform(-4, -2),
        "lrf": random.uniform(0.1, 0.3),
        "momentum": random.uniform(0.85, 0.98),
        "weight_decay": 10 ** random.uniform(-5, -3),
        "warmup_epochs": random.uniform(1, 5),
        "box": random.uniform(5.0, 10.0),
        "cls": random.uniform(0.3, 1.0),
        "dfl": random.uniform(0.5, 2.0),
        "hsv_h": random.uniform(0.0, 0.1),
        "hsv_s": random.uniform(0.3, 0.9),
        "hsv_v": random.uniform(0.3, 0.9),
        "degrees": random.uniform(0.0, 10.0),
        "translate": random.uniform(0.0, 0.2),
        "scale": random.uniform(0.5, 1.0),
        "fliplr": random.uniform(0.3, 0.7),
    }

# ===============================
# RUN YOLO TRAINING
# ===============================
def run_trial(trial_id, hyp):
    hyp_path = f"hyp_trial_{trial_id}.yaml"

    with open(hyp_path, "w") as f:
        yaml.dump(hyp, f)

    cmd = [
        "python", "train.py",
        "--data", DATA_YAML,
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH),
        "--epochs", str(EPOCHS),
        "--weights", MODEL,
        "--device", str(DEVICE),
        "--hyp", hyp_path,
        "--project", PROJECT_DIR,
        "--name", f"trial_{trial_id}",
        "--exist-ok"
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL)

    return hyp_path

# ===============================
# READ METRICS
# ===============================
def get_map(trial_id):
    results_path = Path(PROJECT_DIR) / f"trial_{trial_id}" / "results.csv"
    if not results_path.exists():
        return 0.0

    with open(results_path) as f:
        lines = f.readlines()

    last = lines[-1].split(",")
    map50_95 = float(last[6])  # mAP@50:95
    return map50_95

# ===============================
# MAIN TUNER
# ===============================
def main():
    best_map = 0
    best_hyp = None
    history = []

    for trial in range(N_TRIALS):
        print(f"\n🚀 Trial {trial+1}/{N_TRIALS}")

        hyp = sample_hyperparameters()
        run_trial(trial, hyp)

        score = get_map(trial)
        print(f"📊 mAP@50:95 = {score:.4f}")

        history.append({"trial": trial, "map": score, "hyp": hyp})

        if score > best_map:
            best_map = score
            best_hyp = hyp
            print("🔥 New BEST model found!")

    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_hyp, f, indent=2)

    print("\n🏆 BEST RESULT")
    print(f"mAP@50:95 = {best_map:.4f}")
    print(best_hyp)

if __name__ == "__main__":
    main()
