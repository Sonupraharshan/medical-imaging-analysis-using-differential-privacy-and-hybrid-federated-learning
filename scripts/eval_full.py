#!/usr/bin/env python3
"""
Evaluate a saved global model (PyTorch state_dict) on a folder of test bags.

Usage:
  python scripts/eval_full.py --model artifacts/global_model_round_5.pt --test_bags data/synth_bags
"""

import argparse
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Adjust import paths if necessary
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in os.sys.path:
    os.sys.path.insert(0, SRC)

# Import project model and dataset
from mem_model import MEM
from dataset import BagDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(state_path):
    model = MEM().to(DEVICE)
    state = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate_model(model, test_folder):
    ds = BagDataset(test_folder)
    from torch.utils.data import DataLoader

    def collate(batch):
        # batch: list of (feats (np.ndarray), label)
        feats = [torch.tensor(x[0], dtype=torch.float32) for x in batch]
        labels = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
        maxN = max([f.shape[0] for f in feats])
        D = feats[0].shape[1]
        padded = torch.zeros((len(feats), maxN, D), dtype=torch.float32)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0], :] = f
        return padded, labels

    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    y_true = []
    y_score = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())

    y_pred = [1 if p >= 0.5 else 0 for p in y_score]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
    }
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved model state dict (pth/pt)")
    p.add_argument("--test_bags", required=True, help="Folder with bag_*.npz test bags")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model file not found: {args.model}")
    if not os.path.exists(args.test_bags):
        raise SystemExit(f"Test bags folder not found: {args.test_bags}")

    print("Loading model:", args.model)
    model = load_model(args.model)
    print("Evaluating on test folder:", args.test_bags)

    metrics = evaluate_model(model, args.test_bags)

    print("\nEvaluation results for model:", args.model)
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
