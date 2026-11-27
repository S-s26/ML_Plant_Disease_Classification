"""End-to-end pipeline orchestrator for Plant Disease Classification."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_dataset import download_plantvillage
from src.data.prepare_splits import prepare_splits
from src.features.build_feature_table import build_feature_table
from src.training.evaluate_all import evaluate_all
from src.training.train_classical_mlp import run_classical_and_mlp
from src.training.train_cnn import train_cnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Plant Disease Classification pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--skip-download", action="store_true", help="Skip GitHub dataset download step")
    parser.add_argument(
        "--skip-binary-cnn",
        action="store_true",
        help="Skip dedicated binary Custom CNN (still runs multiclass Custom + Transfer CNN + Binary Rollup)",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_plantvillage(args.config)

    prepare_splits(args.config)
    build_feature_table(args.config)
    run_classical_and_mlp(args.config)

    if not args.skip_binary_cnn:
        train_cnn(arch="custom", task="binary", config_path=args.config)

    train_cnn(arch="custom", task="multiclass", config_path=args.config)
    train_cnn(arch="transfer", task="multiclass", config_path=args.config)
    evaluate_all(args.config)


if __name__ == "__main__":
    main()
