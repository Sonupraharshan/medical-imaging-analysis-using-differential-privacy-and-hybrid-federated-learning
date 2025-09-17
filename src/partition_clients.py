#!/usr/bin/env python3
import os
import argparse
import numpy as np
from glob import glob
import shutil

def iid_split(bags, n_clients):
    np.random.shuffle(bags)
    splits = [bags[i::n_clients] for i in range(n_clients)]
    return splits

def noniid_by_label(bags, n_clients, alpha=0.7):
    import numpy as np
    label0 = [b for b in bags if int(np.load(b)['label'])==0]
    label1 = [b for b in bags if int(np.load(b)['label'])==1]
    splits = [[] for _ in range(n_clients)]
    # assign more label1 to first clients
    for i,b in enumerate(label0):
        splits[i % n_clients].append(b)
    for i,b in enumerate(label1):
        target = i % max(1, int(n_clients*alpha))
        splits[target].append(b)
    return splits

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bags_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_clients", type=int, default=2)
    p.add_argument("--mode", choices=["iid","noniid_labels"], default="iid")
    p.add_argument("--alpha", type=float, default=0.7)
    args = p.parse_args()

    bags = sorted(glob(os.path.join(args.bags_dir, "bag_*.npz")))
    if len(bags)==0:
        print("No bag files in", args.bags_dir); return
    if args.mode=="iid":
        splits = iid_split(bags, args.n_clients)
    else:
        splits = noniid_by_label(bags, args.n_clients, alpha=args.alpha)
    os.makedirs(args.out_dir, exist_ok=True)
    for i, s in enumerate(splits):
        d = os.path.join(args.out_dir, f"client_{i}")
        os.makedirs(d, exist_ok=True)
        for f in s:
            shutil.copy(f, d)
        print(f"client_{i}: {len(s)} bags -> {d}")

if __name__ == "__main__":
    main()
