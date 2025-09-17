#!/usr/bin/env python3
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))

def generate_synthetic_bags(out_dir=os.path.join(ROOT,"data","synth_bags"), n_bags=30, min_inst=8, max_inst=32, feat_dim=1024, seed=0):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_bags):
        N = np.random.randint(min_inst, max_inst+1)
        feats = np.random.randn(N, feat_dim).astype("float32")
        # Simple rule to create learnable labels:
        lab = int(feats[:,0].mean() > 0.0)
        np.savez_compressed(os.path.join(out_dir, f"bag_{i}.npz"), feats=feats, label=lab)
    print("Generated", n_bags, "synthetic bags at", out_dir)

if __name__ == "__main__":
    generate_synthetic_bags()
