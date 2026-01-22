# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import kagglehub
from loguru import logger

from models.bos_model.ufld_v2.demo import model_config as cfg


# ===== Config classes (minimal change, remove globals) =====
class TuSimpleDownloaderConfig:
    DATASET_PATH = "manideep1108/tusimple"
    # relative inside kagglehub version dir
    REQUIRED_TEST_SET = "TUSimple/test_set"


class CULaneDownloaderConfig:
    DATASET_PATH = "manideep1108/culane"
    REQUIRED_ROOT = "CULane"
    REQUIRED_SUBFOLDERS = [
        "driver_100_30frame",
        "driver_193_90frame",
        "driver_37_30frame",
        "list",
    ]


def ensure_symlink(src: str, dst: str) -> None:
    """
    Create a symbolic link dst -> src if dst does not already exist.
    If dst exists (file, directory, or symlink), do nothing.
    """
    if os.path.islink(dst) or os.path.exists(dst):
        logger.info(f"Already exists, skipping: {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    logger.info(f"Created symlink: {dst} -> {src}")


def latest_kagglehub_version_dir(dataset_path: str) -> str | None:
    """
    Return the latest local kagglehub cache version directory without downloading.
    Example:
      ~/.cache/kagglehub/datasets/<owner>/<dataset>/versions/<N>
    """
    owner, ds = dataset_path.split("/", 1)
    base = os.path.expanduser(f"~/.cache/kagglehub/datasets/{owner}/{ds}/versions")
    if not os.path.isdir(base):
        return None

    versions = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full) and name.isdigit():
            versions.append(int(name))

    if not versions:
        return None

    return os.path.join(base, str(max(versions)))


def culane_cache_is_ready(version_dir: str) -> bool:
    """
    Check whether CULane cache contains the minimum required structure.
    """
    root = os.path.join(version_dir, CULaneDownloaderConfig.REQUIRED_ROOT)
    if not os.path.isdir(root):
        return False
    for name in CULaneDownloaderConfig.REQUIRED_SUBFOLDERS:
        if not os.path.isdir(os.path.join(root, name)):
            return False
    return True


def download_tusimple(tusimple_data_dir: str) -> None:
    """Download and link TuSimple dataset."""
    logger.info("Starting TuSimple dataset download...")

    t_latest = latest_kagglehub_version_dir(TuSimpleDownloaderConfig.DATASET_PATH)
    if t_latest is not None and os.path.isdir(os.path.join(t_latest, TuSimpleDownloaderConfig.REQUIRED_TEST_SET)):
        path = t_latest
        logger.info(f"TuSimple cache found (no download): {path}")
    else:
        path = kagglehub.dataset_download(TuSimpleDownloaderConfig.DATASET_PATH)

    os.makedirs(tusimple_data_dir, exist_ok=True)

    source_test_set = os.path.join(path, TuSimpleDownloaderConfig.REQUIRED_TEST_SET)
    if not os.path.isdir(source_test_set):
        raise RuntimeError(f"Missing TuSimple folder: {source_test_set}")

    # Link test_set into tusimple_data_dir/tusimple (same as original semantics)
    tusimple_target = os.path.join(tusimple_data_dir, "tusimple")
    ensure_symlink(source_test_set, tusimple_target)
    logger.info(f"Successfully linked '{TuSimpleDownloaderConfig.REQUIRED_TEST_SET}' to '{tusimple_target}'")


def download_culane(culane_data_dir: str) -> None:
    """Download and link CULane dataset."""
    logger.info("Starting CULane dataset download...")

    c_latest = latest_kagglehub_version_dir(CULaneDownloaderConfig.DATASET_PATH)
    if c_latest is not None and culane_cache_is_ready(c_latest):
        culane_path = c_latest
        logger.info(f"CULane cache found (no download): {culane_path}")
    else:
        culane_path = kagglehub.dataset_download(CULaneDownloaderConfig.DATASET_PATH)

    os.makedirs(culane_data_dir, exist_ok=True)

    source_root = os.path.join(culane_path, CULaneDownloaderConfig.REQUIRED_ROOT)
    if not os.path.isdir(source_root):
        raise RuntimeError(f"Missing CULane folder: {source_root}")

    missing = [
        name
        for name in CULaneDownloaderConfig.REQUIRED_SUBFOLDERS
        if not os.path.isdir(os.path.join(source_root, name))
    ]
    if missing:
        raise RuntimeError(
            "CULane dataset structure is incomplete. Missing:\n  - "
            + "\n  - ".join(missing)
            + f"\nUnder: {source_root}"
        )

    # Link CULane root into culane_data_dir/culane (same as original semantics)
    culane_target = os.path.join(culane_data_dir, "culane")
    ensure_symlink(source_root, culane_target)
    logger.info(f"Successfully linked '{CULaneDownloaderConfig.REQUIRED_ROOT}' to '{culane_target}'")


def main():
    image_data_root = os.path.join(cfg.data_root, "image_data")

    parser = argparse.ArgumentParser(description="Download TuSimple and/or CULane datasets")
    parser.add_argument("--tusimple", action="store_true", help="Download TuSimple dataset")
    parser.add_argument("--culane", action="store_true", help="Download CULane dataset")

    # Argumentize hardcoded paths (defaults keep original behavior)
    parser.add_argument(
        "--tusimple_data_dir",
        default=image_data_root,
        help="TuSimple dataset directory (tusimple symlink will be placed under this dir)",
    )
    parser.add_argument(
        "--culane_data_dir",
        default=image_data_root,
        help="CULane dataset directory (culane symlink will be placed under this dir)",
    )

    args = parser.parse_args()

    # If no dataset flags provided, download both (backward compatibility)
    if not args.tusimple and not args.culane:
        logger.info("No dataset specified, downloading both TuSimple and CULane...")
        download_tusimple(args.tusimple_data_dir)
        download_culane(args.culane_data_dir)
    else:
        if args.tusimple:
            download_tusimple(args.tusimple_data_dir)
        if args.culane:
            download_culane(args.culane_data_dir)

    logger.info("Dataset download completed successfully!")


if __name__ == "__main__":
    main()
