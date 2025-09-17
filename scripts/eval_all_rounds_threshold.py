#!/usr/bin/env python3
"""
Evaluate all saved global_model_round_*.pt files at a fixed threshold (default 0.4850).
Also prints AUC (threshold-free) and saves CSV + a line-plot of metrics vs round.

Usage:
  python scripts/eval_all_rounds_threshold.py --test data/synth_bags --threshold 0.4850 --artifacts artifacts
"""
import os
import argparse
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mem_model import MEM
from dataset import BagDataset

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
    p.add_argument("--artifacts", default="artifacts", help="Artifacts folder containing global_model_round_*.pt")
    p.add_argument("--test", required=True, help="Test bags folder (e.g. data/synth_bags)")
    p.add_argument("--threshold", type=float, default=0.4850, help="Fixed probability threshold to use for classification")
    args = p.parse_args()

    if not os.path.exists(args.artifacts):
        raise SystemExit(f"Artifacts folder not found: {args.artifacts}")
    if not os.path.exists(args.test):
        raise SystemExit(f"Test folder not found: {args.test}")

    files = sorted([f for f in os.listdir(args.artifacts) if f.startswith("global_model_round_") and f.endswith(".pt")])
    if not files:
        raise SystemExit("No global_model_round_*.pt files found in artifacts folder.")

    rows = []
    rounds = []
    for fn in files:
        path = os.path.join(args.artifacts, fn)
        try:
            round_num = int(fn.split("_")[-1].split(".")[0])
        except:
            continue
        model = load_model(path)
        y_true, y_score = compute_scores(model, args.test)
        preds = (y_score >= args.threshold).astype(int)
        auc = (roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"))
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append({
            "round": round_num,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "n": len(y_true)
        })
        rounds.append(round_num)

    rows = sorted(rows, key=lambda x: x["round"])

    # CSV
    out_csv = os.path.join(args.artifacts, f"metrics_by_round_threshold_{args.threshold:.4f}.csv")
    with open(out_csv, "w", newline="") as fh:
        fh.write("round,accuracy,precision,recall,f1,auc,n\n")
        for r in rows:
            fh.write(f'{r["round"]},{r["accuracy"]:.6f},{r["precision"]:.6f},{r["recall"]:.6f},{r["f1"]:.6f},{r["auc"]:.6f},{r["n"]}\n')

    print("Saved CSV ->", out_csv)
    print("round,accuracy,precision,recall,f1,auc,n")
    for r in rows:
        print(f'{r["round"]},{r["accuracy"]:.6f},{r["precision"]:.6f},{r["recall"]:.6f},{r["f1"]:.6f},{r["auc"]:.6f},{r["n"]}')

    # Plot metrics vs rounds
    rounds = [r["round"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    precs = [r["precision"] for r in rows]
    recs = [r["recall"] for r in rows]
    f1s = [r["f1"] for r in rows]
    aucs = [r["auc"] for r in rows]

    plt.figure(figsize=(10,6))
    plt.plot(rounds, accs, marker="o", label="Accuracy")
    plt.plot(rounds, aucs, marker="o", label="AUC")
    plt.plot(rounds, f1s, marker="o", label="F1")
    plt.plot(rounds, precs, marker="o", label="Precision")
    plt.plot(rounds, recs, marker="o", label="Recall")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title(f"Metrics by Round (threshold={args.threshold:.4f})")
    plt.legend()
    plt.grid(True)
    out_png = os.path.join(args.artifacts, f"metrics_rounds_threshold_{args.threshold:.4f}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("Saved plot ->", out_png)

if __name__ == "__main__":
    main()
