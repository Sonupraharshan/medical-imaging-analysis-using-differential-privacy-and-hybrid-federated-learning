#!/usr/bin/env python3
"""
Compute per-round metrics, find best thresholds, and plot metrics/curves.

Usage:
  python scripts/threshold_and_plots.py --artifacts artifacts --test data/synth_bags

Options:
  --round INT     : evaluate this specific round (e.g. 5). If omitted, script picks the round with highest AUC.
  --out_dir PATH  : where to save CSV and plots (default: artifacts)
"""
import os
import argparse
import sys
import glob
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv

# Ensure repo src on path
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

def get_scores_for_model(model, test_folder):
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
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_score)

def metrics_from_scores(y_true, y_score, threshold=0.5):
    preds = (y_score >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "n": len(y_true)}

def sweep_thresholds(y_true, y_score, n_steps=101):
    ts = np.linspace(0.0, 1.0, n_steps)
    metrics = []
    best_f1 = (-1, None)
    best_youden = (-1, None)
    for t in ts:
        m = metrics_from_scores(y_true, y_score, threshold=t)
        metrics.append((t, m))
        if m["f1"] > best_f1[0]:
            best_f1 = (m["f1"], t)
        # Youden = tpr - fpr
        try:
            fpr, tpr, _ = roc_curve(y_true, (y_score >= t).astype(int))
            youden = tpr[1] - fpr[1] if len(tpr) > 1 else tpr[0] - fpr[0]
        except Exception:
            youden = float("-inf")
        if youden > best_youden[0]:
            best_youden = (youden, t)
    return metrics, best_f1, best_youden

def plot_metrics_by_round(rows, outpath):
    # rows: list of dicts with keys: round, accuracy, precision, recall, f1, auc
    rounds = [r["round"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]
    auc = [r["auc"] for r in rows]

    plt.figure(figsize=(10,6))
    plt.plot(rounds, acc, marker="o", label="Accuracy")
    plt.plot(rounds, auc, marker="o", label="AUC")
    plt.plot(rounds, f1, marker="o", label="F1")
    plt.plot(rounds, prec, marker="o", label="Precision")
    plt.plot(rounds, rec, marker="o", label="Recall")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title("Metrics by Round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_curves_for_round(y_true, y_score, metrics_sweep, best_f1, best_youden, outpath):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # PR
    prec_vals, rec_vals, pr_thresh = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score) if len(set(y_true))>1 else float("nan")
    auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else float("nan")

    fig, axs = plt.subplots(1,3, figsize=(18,5))

    axs[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axs[0].plot([0,1],[0,1], linestyle="--")
    axs[0].set_xlabel("FPR")
    axs[0].set_ylabel("TPR")
    axs[0].set_title("ROC Curve")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(rec_vals, prec_vals, label=f"AP={ap:.3f}")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_title("Precision-Recall Curve")
    axs[1].legend()
    axs[1].grid(True)

    # Metric vs threshold
    ts = [m[0] for m in metrics_sweep]
    accs = [m[1]["accuracy"] for m in metrics_sweep]
    precs = [m[1]["precision"] for m in metrics_sweep]
    recs = [m[1]["recall"] for m in metrics_sweep]
    f1s = [m[1]["f1"] for m in metrics_sweep]

    axs[2].plot(ts, accs, marker=".", label="Accuracy")
    axs[2].plot(ts, precs, marker=".", label="Precision")
    axs[2].plot(ts, recs, marker=".", label="Recall")
    axs[2].plot(ts, f1s, marker=".", label="F1")
    axs[2].axvline(best_f1[1], linestyle="--", label=f"F1-best={best_f1[1]:.2f}")
    axs[2].axvline(best_youden[1], linestyle=":", label=f"Youden={best_youden[1]:.2f}")
    axs[2].set_xlabel("Threshold")
    axs[2].set_ylabel("Metric")
    axs[2].set_title("Metrics vs Threshold")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    fig.suptitle("ROC / PR / Threshold sweep", y=1.02)
    plt.savefig(outpath)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--test", required=True)
    p.add_argument("--round", type=int, default=None, help="Round to analyze in-depth (default: pick highest AUC)")
    p.add_argument("--out_dir", default=None, help="Where to save outputs; defaults to artifacts")
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else args.artifacts
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.artifacts, "global_model_round_*.pt")))
    if not files:
        raise SystemExit("No checkpoints found in artifacts folder.")

    rows = []
    all_scores = {}
    for fp in files:
        fn = os.path.basename(fp)
        round_num = int(fn.split("_")[-1].split(".")[0])
        model = load_model(fp)
        y_true, y_score = get_scores_for_model(model, args.test)
        all_scores[round_num] = (y_true, y_score)
        m = metrics_from_scores(y_true, y_score, threshold=0.5)
        m["round"] = round_num
        rows.append(m)

    # sort rows by round
    rows = sorted(rows, key=lambda x: x["round"])

    # save CSV
    csv_path = os.path.join(out_dir, "metrics_by_round.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["round","accuracy","precision","recall","f1","auc","n"])
        for r in rows:
            writer.writerow([r["round"], r["accuracy"], r["precision"], r["recall"], r["f1"], r["auc"], r["n"]])

    # plot metrics vs rounds
    plot_metrics_by_round(rows, os.path.join(out_dir, "metrics_rounds.png"))
    print("Saved metrics CSV ->", csv_path)
    print("Saved metrics-by-round plot ->", os.path.join(out_dir, "metrics_rounds.png"))

    # choose round for deeper analysis
    if args.round is None:
        # pick round with max AUC (ignore nan)
        sorted_by_auc = sorted([r for r in rows if not np.isnan(r["auc"])], key=lambda x: x["auc"], reverse=True)
        if not sorted_by_auc:
            chosen = rows[-1]["round"]
        else:
            chosen = sorted_by_auc[0]["round"]
    else:
        chosen = args.round

    print("Chosen round for deep analysis:", chosen)
    y_true, y_score = all_scores[chosen]

    # threshold sweep
    metrics_sweep, best_f1, best_youden = sweep_thresholds(y_true, y_score, n_steps=201)
    print("Best F1:", best_f1)
    print("Best Youden (tpr-fpr):", best_youden)

    # save threshold sweep CSV
    ts_csv = os.path.join(out_dir, f"metrics_round_{chosen}_thresholds.csv")
    with open(ts_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold","accuracy","precision","recall","f1"])
        for t, m in metrics_sweep:
            writer.writerow([t, m["accuracy"], m["precision"], m["recall"], m["f1"]])
    print("Saved threshold sweep CSV ->", ts_csv)

    # plots for chosen round
    plot_curves_for_round(y_true, y_score, metrics_sweep, best_f1, best_youden, os.path.join(out_dir, f"metrics_round_{chosen}_curves.png"))
    print("Saved ROC/PR/threshold-plot ->", os.path.join(out_dir, f"metrics_round_{chosen}_curves.png"))

if __name__ == "__main__":
    main()
