#!/usr/bin/env python3
import argparse, os
from client import MemNumPyClient
import flwr as fl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server_address", default="localhost:8080")
    p.add_argument("--client_data", default=None)
    p.add_argument("--target_epsilon", type=float, default=5.0)
    p.add_argument("--target_delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--local_epochs", type=int, default=1)
    args = p.parse_args()

    client_data = args.client_data or os.environ.get("CLIENT_DATA")
    if client_data is None:
        raise ValueError("Provide client data folder via --client_data or CLIENT_DATA env var.")
    client = MemNumPyClient(client_folder=client_data,
                            local_epochs=args.local_epochs,
                            target_epsilon=args.target_epsilon,
                            target_delta=args.target_delta,
                            max_grad_norm=args.max_grad_norm)
    print("Starting Flower client for data:", client_data)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
