#!/usr/bin/env python3
"""
Flower server: saves global model per round and evaluates on test set (accuracy + AUC).
Robust to Flower returning either fl.common.Parameters or plain ndarray lists.
"""
import argparse
import os
import torch
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import BagDataset
from mem_model import MEM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def params_to_ndlist(params):
    """
    Convert Flower Parameters object OR plain list into a list of numpy ndarrays.
    """
    # If it's a Flower Parameters object, convert using helper
    try:
        if isinstance(params, fl.common.Parameters):
            nds = parameters_to_ndarrays(params)
            return nds
    except Exception:
        # fall through and try other heuristics
        pass

    # If it already looks like a list/tuple of numpy arrays, return as-is
    if isinstance(params, (list, tuple)):
        return list(params)

    # If it's a dict-like mapping of name->ndarray
    if hasattr(params, "items"):
        return [v for _, v in params.items()]

    # Otherwise, try to coerce to list
    return list(params)


def save_model_from_params(params, save_path):
    """
    Save a PyTorch state_dict constructed from a list of ndarrays or a fl Parameters object.
    Matching is done in the order of model.state_dict().keys().
    """
    nds = params_to_ndlist(params)

    model = MEM().to(DEVICE)
    state = model.state_dict()

    # If length mismatch, try a safer mapping if possible
    if len(nds) != len(list(state.keys())):
        # Attempt to handle the case where Flower wraps weights differently:
        # If nds is a single ndarray of dtype object containing arrays, try to unwrap
        if len(nds) == 1 and hasattr(nds[0], "__len__") and not isinstance(nds[0], np.ndarray):
            try:
                nds = list(nds[0])
            except Exception:
                pass

    # Map ndarrays (in order) to the state_dict keys
    for k, v in zip(state.keys(), nds):
        try:
            state[k] = torch.tensor(v)
        except Exception:
            # as a fallback, if v is a fl.common.NDArray wrapper, try to convert via numpy
            try:
                arr = np.array(v)
                state[k] = torch.tensor(arr)
            except Exception as e:
                raise RuntimeError(f"Failed to convert parameter for key {k}: {e}")

    model.load_state_dict(state)
    torch.save(model.state_dict(), save_path)
    print("Saved global model ->", save_path)


def get_evaluate_fn(test_bags_folder):
    """Return an evaluation function for Flower that evaluates accuracy and AUC on test set."""
    if not test_bags_folder or not os.path.exists(test_bags_folder):
        return None

    test_ds = BagDataset(test_bags_folder)

    def evaluate(weights):
        # Convert weights (could be fl Parameters or list)
        nds = params_to_ndlist(weights)

        # Build model and set weights
        model = MEM().to(DEVICE)
        state = model.state_dict()

        for k, v in zip(state.keys(), nds):
            state[k] = torch.tensor(v)
        model.load_state_dict(state)
        model.eval()

        y_true = []
        y_score = []
        with torch.no_grad():
            from torch.utils.data import DataLoader

            def collate(batch):
                feats = [torch.tensor(x[0]) for x in batch]
                labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
                maxN = max([f.shape[0] for f in feats])
                D = feats[0].shape[1]
                padded = torch.zeros((len(feats), maxN, D), dtype=torch.float32)
                for i, f in enumerate(feats):
                    padded[i, : f.shape[0], :] = f
                return padded, labels

            loader = DataLoader(test_ds, batch_size=8, collate_fn=collate)
            for X, y in loader:
                X = X.to(DEVICE)
                logits = model(X)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_score.extend(probs.tolist())
                y_true.extend(y.numpy().tolist())

        if len(set(y_true)) < 2:
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                auc = float("nan")

        acc = float(accuracy_score(y_true, [1 if s >= 0.5 else 0 for s in y_score]))
        return 0.0, {"accuracy": acc, "auc": auc}

    return evaluate


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves the global model after each round."""

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            # aggregated is typically a tuple (Parameters, num_examples) or similar
            params = aggregated[0]
            save_path = os.path.join(ARTIFACTS_DIR, f"global_model_round_{rnd}.pt")
            save_model_from_params(params, save_path)
        return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--test_bags", default=None)
    args = parser.parse_args()

    # Create strategy with evaluation function if test set is provided
    strategy = SaveModelStrategy(
        fraction_fit=0.5, min_fit_clients=1, min_available_clients=1, evaluate_fn=get_evaluate_fn(args.test_bags)
    )

    # Compatibility workaround: Flower's compatibility layer in some versions expects
    # a config object with attributes like num_rounds and round_timeout.
    server_config = type("ServerConfig", (), {"num_rounds": args.rounds, "round_timeout": None})()

    print("Starting Flower server at", args.server_address)
    fl.server.start_server(server_address=args.server_address, config=server_config, strategy=strategy)


if __name__ == "__main__":
    main()
