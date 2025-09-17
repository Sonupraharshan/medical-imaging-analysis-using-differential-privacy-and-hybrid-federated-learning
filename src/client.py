#!/usr/bin/env python3
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mem_model import MEM  # alias, see client_runner.py imports
from dataset import BagDataset
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), parameters):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)

class MemNumPyClient(fl.client.NumPyClient):
    def __init__(self, client_folder, local_epochs=1, batch_size=4, lr=1e-3,
                 target_epsilon=5.0, target_delta=1e-5, max_grad_norm=1.0):
        self.model = MEM().to(DEVICE)
        self.client_folder = client_folder
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.dataset = BagDataset(client_folder)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # pad to max instances in batch
        feats = [torch.tensor(x[0]) for x in batch]
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
        maxN = max([f.shape[0] for f in feats])
        D = feats[0].shape[1]
        padded = torch.zeros((len(feats), maxN, D), dtype=torch.float32)
        for i,f in enumerate(feats):
            padded[i,:f.shape[0],:] = f
        return padded, labels

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        # Make private
        privacy_engine = PrivacyEngine()
        try:
            self.model, optimizer, self.loader = privacy_engine.make_private_with_epsilon(
                module=self.model, optimizer=optimizer, data_loader=self.loader,
                epochs=self.local_epochs, target_epsilon=self.target_epsilon,
                target_delta=self.target_delta, max_grad_norm=self.max_grad_norm)
        except Exception as e:
            print("Opacus make_private error:", e)
        self.model.train()
        for epoch in range(self.local_epochs):
            for X, y in self.loader:
                X = X.to(DEVICE); y = y.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        return get_parameters(self.model), len(self.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for X, y in self.loader:
                X = X.to(DEVICE); y = y.to(DEVICE)
                logits = self.model(X)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total if total>0 else 0.0
        return float(1.0 - acc), total, {"accuracy": float(acc)}
