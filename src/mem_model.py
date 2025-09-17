#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBlock(nn.Module):
    def __init__(self, feat_dim, mem_dim, num_units):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_units, mem_dim))
        self.key = nn.Linear(feat_dim, mem_dim)
        self.value = nn.Linear(feat_dim, mem_dim)
        self.out = nn.Linear(mem_dim, mem_dim)

    def forward(self, X):
        # X: (B, N, feat_dim)
        K = self.key(X)  # (B,N,mem_dim)
        mem = self.mem.unsqueeze(0)  # (1,U,mem_dim)
        scores = torch.einsum("bnm,bum->bnu", K, mem)  # (B,N,U)
        attn = torch.softmax(scores, dim=2)
        V = self.value(X)  # (B,N,mem_dim)
        mem_out = torch.einsum("bnu,bnm->bum", attn, V)  # (B,U,mem_dim)
        out = mem_out.mean(dim=1)  # (B,mem_dim)
        return F.relu(self.out(out))

class MEM(nn.Module):
    def __init__(self, feat_dim=1024, mem_dim=512, num_units=16, num_blocks=2, hidden=256, n_classes=2):
        super().__init__()
        self.blocks = nn.ModuleList([MemoryBlock(feat_dim if i==0 else mem_dim, mem_dim, num_units) for i in range(num_blocks)])
        self.classifier = nn.Sequential(nn.Linear(mem_dim, hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, n_classes))

    def forward(self, X):
        # X: (B, N, feat_dim)
        out = X
        for i, blk in enumerate(self.blocks):
            pooled = blk(out)   # (B, mem_dim)
            out = pooled.unsqueeze(1).repeat(1, X.size(1), 1)
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
