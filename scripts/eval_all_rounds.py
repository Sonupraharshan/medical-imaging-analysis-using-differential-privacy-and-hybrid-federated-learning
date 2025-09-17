#!/usr/bin/env python3
"""
Evaluate all saved global_model_round_*.pt files and print CSV-style metrics.

Usage:
  python scripts/eval_all_rounds.py --artifacts artifacts --test data/synth_bags
"""
import os
import argparse
import sys
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mem_model import MEM
from dataset import BagDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_model(path):
    model = MEM()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def compute_scores(model, test_folder):
    ds = BagDataset(test_folder)

    def collate(batch):
        feats = [torch.tensor(x[0], dtype=torch.float32) for x in batch]
        labels = [int(x[1]) for x in batch]
        maxN = max([f.shape[0] for f in feats])
        D = feats[0].shape[1]
        padded = torch.zeros((len(feats), maxN, D), dtype=torch.float32)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0], :] = f
        return padded, torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    y_true = []
    y_score = []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_score)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts", help="Folder with global_model_round_*.pt")
    p.add_argument("--test", required=True, help="Test bags folder (e.g. data/synth_bags)")
    args = p.parse_args()

    if not os.path.exists(args.artifacts):
        raise SystemExit(f"Artifacts folder not found: {args.artifacts}")
    if not os.path.exists(args.test):
        raise SystemExit(f"Test folder not found: {args.test}")

    files = sorted([f for f in os.listdir(args.artifacts) if f.startswith("global_model_round_") and f.endswith(".pt")])
    if not files:
        raise SystemExit("No global_model_round_*.pt files found in artifacts folder.")

    print("round,accuracy,precision,recall,f1,auc,n")
    for fn in files:
        path = os.path.join(args.artifacts, fn)
        model = load_model(path)
        y_true, y_score = compute_scores(model, args.test)
        preds = (y_score >= 0.5).astype(int)
        auc = (roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"))
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        print(f"{fn.replace('.pt','')},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{auc:.4f},{len(y_true)}")

if __name__ == "__main__":
    main()
