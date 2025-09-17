#!/usr/bin/env python3
"""
Orchestrator: run_pipeline.py

Usage (synthetic quick test):
  python src/run_pipeline.py --mode synth --n_clients 2 --rounds 5

Usage (real, after placing WSIs into data/wsis and labels.csv):
  python src/run_pipeline.py --mode real --wsi_dir data/wsis --labels_csv data/labels.csv \
    --n_clients 4 --rounds 250 --patch_size 1000 --n_clusters 50 --sample_frac 0.10
"""
import argparse
import os
import time
import subprocess
import signal
import shutil
import socket

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")

def run(cmd, env=None):
    print("RUN >", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)

def ensure_dirs():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "clients"), exist_ok=True)

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    addr, port = s.getsockname()
    s.close()
    return port

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synth","real"], default="synth")
    parser.add_argument("--wsi_dir", default=os.path.join(DATA_DIR,"wsis"))
    parser.add_argument("--labels_csv", default=os.path.join(DATA_DIR,"labels.csv"))
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--sample_frac", type=float, default=0.10)
    args = parser.parse_args()

    ensure_dirs()

    # 1) Data prep
    if args.mode == "synth":
        print("=== Generating synthetic data ===")
        subprocess.run(["python", os.path.join(SRC,"synth_data.py")], check=True)
        subprocess.run(["python", os.path.join(SRC,"partition_clients.py"),
                        "--bags_dir", os.path.join(ROOT,"data","synth_bags"),
                        "--out_dir", os.path.join(ROOT,"data","clients"),
                        "--n_clients", str(args.n_clients),
                        "--mode", "iid"], check=True)
    else:
        print("=== Running WSI -> bag preparation (may take long) ===")
        cmd = ["python", os.path.join(SRC,"prepare_bags.py"),
               "--wsi_dir", args.wsi_dir,
               "--out_dir", os.path.join(ROOT,"data","bags"),
               "--tmp_dir", os.path.join(ROOT,"tmp_patches"),
               "--labels_csv", args.labels_csv,
               "--patch_size", str(args.patch_size),
               "--n_clusters", str(args.n_clusters),
               "--sample_frac", str(args.sample_frac)]
        subprocess.run(cmd, check=True)
        # partition to clients (n_clients)
        subprocess.run(["python", os.path.join(SRC,"partition_clients.py"),
                        "--bags_dir", os.path.join(ROOT,"data","bags"),
                        "--out_dir", os.path.join(ROOT,"data","clients"),
                        "--n_clients", str(args.n_clients),
                        "--mode", "noniid_labels"], check=True)

    # 2) Start Flower server (as subprocess) on a free port
    server_port = find_free_port()
    server_addr = f"0.0.0.0:{server_port}"
    print(f"Starting Flower server on port {server_port}")
    server_proc = run([
        "python",
        os.path.join(SRC, "server.py"),
        "--server_address", server_addr,
        "--rounds", str(args.rounds),
        "--test_bags", os.path.join(ROOT, "data", "bags_test")
    ])
    time.sleep(2)

    # Wait until the server port is accepting connections (timeout after 15s)
    import socket
    timeout = 15.0
    start_t = time.time()
    server_ready = False
    while time.time() - start_t < timeout:
        try:
            with socket.create_connection(("localhost", server_port), timeout=1):
                server_ready = True
                break
        except Exception:
            time.sleep(0.5)
    if not server_ready:
        print(f"Warning: server did not open port {server_port} within {timeout} seconds. Clients may fail to connect.")
    else:
        print(f"Server is listening on port {server_port}; launching clients.")


    # 3) Launch clients pointing to the same auto-picked port
    client_addr = f"localhost:{server_port}"
    client_procs = []
    for i in range(args.n_clients):
        client_dir = os.path.join(ROOT, "data", "clients", f"client_{i}")
        if not os.path.exists(client_dir):
            print(f"Warning: client dir not found: {client_dir} (skipping launch)")
            continue
        print(f"Starting client_{i} -> {client_dir}")
        env = os.environ.copy()
        env["CLIENT_DATA"] = client_dir
        env["CLIENT_ID"] = str(i)
        p = run([
            "python",
            os.path.join(SRC, "client_runner.py"),
            "--server_address", client_addr,
            "--target_epsilon", "5.0",
            "--target_delta", "1e-5",
            "--max_grad_norm", "1.0",
            "--local_epochs", "1"
        ], env=env)
        client_procs.append(p)
        time.sleep(0.5)

    try:
        # wait for server to finish (server blocks until rounds complete)
        server_proc.wait()
        print("Server finished.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping processes...")
    finally:
        # cleanup
        for p in client_procs:
            try:
                p.send_signal(signal.SIGINT)
                p.kill()
            except Exception:
                pass
        try:
            server_proc.kill()
        except Exception:
            pass

    print("=== Pipeline finished ===")
    print("Final artifacts (bags, clients) are under ./data/")

if __name__ == "__main__":
    main()
