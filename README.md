# Federated DP MEM (repro pipeline)

Run synthetic smoke test:
python src/run_pipeline.py --mode synth --n_clients 2 --rounds 5

Run real TCGA pipeline (after downloading WSIs into data/wsis and creating labels.csv):
python src/run_pipeline.py --mode real --wsi_dir data/wsis --labels_csv data/labels.csv --n_clients 4 --rounds 250

Notes:

- Install system deps (OpenSlide) and Python deps from requirements.txt
- GPU recommended for DenseNet feature extraction and training
- The run_pipeline orchestrates everything: prep -> partition -> server -> clients -> evaluation
