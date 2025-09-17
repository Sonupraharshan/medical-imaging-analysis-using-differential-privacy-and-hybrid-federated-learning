#!/usr/bin/env python3
"""
Find best round by AUC, find the F1-optimal threshold for that round,
print metrics and save a short report to artifacts/report.txt.
Also saves plots (ROC, PR, F1 vs threshold).
"""
import os
import argparse
import sys
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mem_model import MEM
from dataset import BagDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    model = MEM().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def scores_for_model(model, test_folder):
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
    y_true, y_score = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_score)

def metrics_at_threshold(y_true, y_score, t):
    preds = (y_score >= t).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true))>1 else float("nan"),
        "n": int(len(y_true))
    }

def find_best_f1_threshold(y_true, y_score, steps=201):
    ts = np.linspace(0.0, 1.0, steps)
    best = (-1.0, None)
    for t in ts:
        f1 = f1_score(y_true, (y_score>=t).astype(int), zero_division=0)
        if f1 > best[0]:
            best = (f1, float(t))
    return best

def plot_curves(y_true, y_score, best_f1_t, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(outdir,"roc.png"))
    plt.close()

    # Precision–Recall
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(outdir,"pr.png"))
    plt.close()

    # F1 vs threshold
    ts = np.linspace(0,1,201)
    f1s = [f1_score(y_true,(y_score>=t).astype(int),zero_division=0) for t in ts]
    plt.figure()
    plt.plot(ts, f1s, label="F1")
    plt.axvline(best_f1_t, color="r", linestyle="--", label=f"Best t={best_f1_t:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("F1 vs Threshold")
    plt.legend()
    plt.savefig(os.path.join(outdir,"f1_vs_threshold.png"))
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--test", required=True)
    args = p.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    files = [f for f in os.listdir(args.artifacts) if f.startswith("global_model_round_") and f.endswith(".pt")]
    if not files:
        raise SystemExit("No checkpoints found in artifacts folder.")
    rounds = []
    scores_cache = {}
    for fn in files:
        path = os.path.join(args.artifacts, fn)
        try:
            rnd = int(fn.split("_")[-1].split(".")[0])
        except:
            continue
        model = load_model(path)
        y_true, y_score = scores_for_model(model, args.test)
        auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true))>1 else float("nan")
        rounds.append((rnd, path, auc))
        scores_cache[rnd] = (y_true, y_score)

    rounds = sorted(rounds, key=lambda x: x[0])
    rounds_with_auc = [r for r in rounds if not np.isnan(r[2])]
    chosen = max(rounds_with_auc, key=lambda x: x[2])[0] if rounds_with_auc else rounds[-1][0]

    y_true, y_score = scores_cache[chosen]
    best_f1_val, best_f1_t = find_best_f1_threshold(y_true, y_score, steps=201)
    metrics = metrics_at_threshold(y_true, y_score, best_f1_t)

    # save plots
    outdir = os.path.join(args.artifacts, f"round_{chosen}_plots")
    plot_curves(y_true, y_score, best_f1_t, outdir)

    # save text report
    report_lines = []
    report_lines.append(f"Chosen round (best AUC): {chosen}")
    report_lines.append(f"F1-optimal threshold: {best_f1_t:.4f}")
    report_lines.append(f"F1 at that threshold: {best_f1_val:.4f}")
    report_lines.append("Metrics at F1-optimal threshold:")
    for k in ["accuracy","precision","recall","f1","auc","n"]:
        report_lines.append(f"  {k}: {metrics[k]}")
    outpath = os.path.join(args.artifacts, "report.txt")
    with open(outpath, "w") as fh:
        fh.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nSaved report -> {outpath}")
    print(f"Saved plots -> {outdir}")

if __name__ == "__main__":
    main()
