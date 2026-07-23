from pathlib import Path
import random
import shutil


# =========================
# Manual parameters
# =========================

SOURCE_DIR = r"/data/mortal_gbmj/dataset/version3/test/"
TARGET_DIR_NAME = r"/data/mortal_gbmj/dataset/version3/val/"
SAMPLE_COUNT = 2000
RANDOM_SEED = 20260411
RECURSIVE = False
OVERWRITE_EXISTING = False


def main():
    source_dir = Path(SOURCE_DIR)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source dir does not exist: {source_dir}")

    target_dir = source_dir.parent / TARGET_DIR_NAME
    if target_dir.exists() and not target_dir.is_dir():
        raise RuntimeError(f"target path exists and is not a directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if RECURSIVE else "*"
    files = [path for path in source_dir.glob(pattern) if path.is_file()]
    if not files:
        raise RuntimeError(f"no files found in: {source_dir}")

    if SAMPLE_COUNT <= 0:
        raise ValueError("SAMPLE_COUNT must be > 0")
    if SAMPLE_COUNT > len(files):
        raise ValueError(
            f"SAMPLE_COUNT={SAMPLE_COUNT} is larger than available files={len(files)}"
        )

    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(files, SAMPLE_COUNT)

    copied = 0
    skipped = 0
    for src in selected:
        if RECURSIVE:
            rel_path = src.relative_to(source_dir)
            dst = target_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst = target_dir / src.name

        if dst.exists() and not OVERWRITE_EXISTING:
            skipped += 1
            continue

        shutil.move(str(src), str(dst))
        copied += 1

    print(f"source_dir={source_dir}")
    print(f"target_dir={target_dir}")
    print(f"available_files={len(files)}")
    print(f"selected_files={len(selected)}")
    print(f"moved_files={copied}")
    print(f"skipped_existing={skipped}")


if __name__ == "__main__":
    main()
