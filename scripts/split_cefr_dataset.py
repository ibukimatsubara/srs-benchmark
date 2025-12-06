from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into pretrain/valid halves")
    parser.add_argument("--source", required=True, help="Path to original dataset root")
    parser.add_argument(
        "--pretrain-dir",
        default=None,
        help="Output directory for pretrain split (default: <source>_pretrain)",
    )
    parser.add_argument(
        "--valid-dir",
        default=None,
        help="Output directory for valid split (default: <source>_valid)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def propagate_static_dirs(source: Path, targets: Iterable[Path], name: str, copy: bool) -> None:
    src_dir = source / name
    if not src_dir.exists():
        return
    for target in targets:
        dst_dir = target / name
        if dst_dir.exists() or dst_dir.is_symlink():
            continue
        ensure_parent(dst_dir)
        if copy:
            shutil.copytree(src_dir, dst_dir)
        else:
            os.symlink(src_dir, dst_dir, target_is_directory=True)


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    pretrain_dir = (
        Path(args.pretrain_dir).resolve()
        if args.pretrain_dir
        else Path(f"{source}_pretrain").resolve()
    )
    valid_dir = (
        Path(args.valid_dir).resolve()
        if args.valid_dir
        else Path(f"{source}_valid").resolve()
    )

    revlog_root = source / "revlogs"
    if not revlog_root.exists():
        raise SystemExit(f"Revlog directory not found: {revlog_root}")

    user_dirs = sorted(revlog_root.glob("user_id=*"))
    if not user_dirs:
        raise SystemExit("No user directories found under revlogs")

    mid = len(user_dirs) // 2
    split_map = {
        pretrain_dir: user_dirs[:mid],
        valid_dir: user_dirs[mid:],
    }

    for target_root, subset in split_map.items():
        revlog_target = target_root / "revlogs"
        revlog_target.mkdir(parents=True, exist_ok=True)
        for src in subset:
            dst = revlog_target / src.name
            link_or_copy(src, dst, copy=args.copy)

    for name in ("cards", "decks"):
        propagate_static_dirs(source, split_map.keys(), name, args.copy)

    print(
        f"Split {len(user_dirs)} users into {len(split_map[pretrain_dir])} pretrain and "
        f"{len(split_map[valid_dir])} valid entries."
    )
    print(f"Pretrain dir: {pretrain_dir}\nValid dir: {valid_dir}")


if __name__ == "__main__":
    main()
