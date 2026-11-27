"""Download PlantVillage dataset (raw/color + leaf_grouping) from GitHub via sparse checkout."""

import argparse
import subprocess
from pathlib import Path

from src.utils.io import load_config, resolve_path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    """Execute a subprocess command and raise a clear error on failure."""
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def download_plantvillage(config_path: str | None = None, force: bool = False) -> Path:
    """Sparse-clone spMohanty/PlantVillage-Dataset keeping raw/color, leaf_grouping, and README.md."""
    cfg = load_config(config_path)
    repo_url = cfg["dataset"]["repo_url"]
    target_dir = resolve_path(cfg["paths"]["raw_repo_dir"])
    color_dir = resolve_path(cfg["paths"]["raw_color_dir"])

    if color_dir.exists() and any(color_dir.iterdir()) and not force:
        class_dirs = [d for d in color_dir.iterdir() if d.is_dir()]
        print(f"Dataset already exists at {color_dir} ({len(class_dirs)} class folders).")
        return color_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if force and target_dir.exists():
        raise RuntimeError(
            f"Target directory {target_dir} already exists. Delete it manually before passing --force."
        )

    if not (target_dir / ".git").exists():
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir)
        run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(target_dir),
            ]
        )

    run_cmd(
        ["git", "sparse-checkout", "set", "raw/color", "leaf_grouping"],
        cwd=target_dir,
    )

    if not color_dir.exists():
        raise FileNotFoundError(f"Expected directory missing after checkout: {color_dir}")

    class_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    total_images = sum(len(list(d.glob("*"))) for d in class_dirs)
    print(f"Downloaded PlantVillage raw/color: {len(class_dirs)} classes, {total_images} files.")
    return color_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PlantVillage dataset from GitHub.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Fail if directory exists and re-verify")
    args = parser.parse_args()
    download_plantvillage(args.config, force=args.force)


if __name__ == "__main__":
    main()
