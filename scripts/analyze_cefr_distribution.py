"""Summarize CEFR level distribution across user revlogs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Counter as CounterType

import pandas as pd
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config, create_parser


def scan_cefr_levels(root: Path, max_users: int | None) -> tuple[CounterType[int], int, int]:
    user_dirs = sorted(root.glob("user_id=*"))
    if max_users is not None:
        user_dirs = user_dirs[:max_users]
    counts: CounterType[int] = Counter()
    total_reviews = 0
    total_users = 0

    for user_dir in tqdm(user_dirs, desc="Scanning users"):
        df = pd.read_parquet(user_dir)
        cefr = df.get("cefr_level")
        if cefr is None:
            cefr = pd.Series([0] * len(df))
        cefr = cefr.fillna(0).astype(int)
        counts.update(cefr.tolist())
        total_reviews += len(cefr)
        total_users += 1
    return counts, total_reviews, total_users


def main() -> None:
    parser = create_parser()
    parser.add_argument("--max-users", type=int, default=None, help="Limit user count for sampling")
    parser.add_argument("--skip-zero", action="store_true", help="Hide entries with cefr_level=0")
    args = parser.parse_args()
    config = Config(args)

    revlog_root = config.data_path / "revlogs"
    if not revlog_root.exists():
        raise SystemExit(f"Revlog path not found: {revlog_root}")

    counts, total_reviews, total_users = scan_cefr_levels(revlog_root, args.max_users)
    cefr_names = {0: "Not Found", 1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}

    print(f"Data source: {revlog_root}")
    print(f"Users scanned: {total_users}\nTotal reviews: {total_reviews:,}\n")

    grand_total = sum(counts.values())
    if grand_total == 0:
        print("No reviews found.")
        return

    print("CEFR level distribution:")
    for level in range(0, 7):
        if args.skip_zero and level == 0:
            continue
        value = counts.get(level, 0)
        pct = value / grand_total * 100
        label = cefr_names.get(level, str(level))
        print(f"  {label:9s}: {value:>12,} ({pct:6.2f}%)")


if __name__ == "__main__":
    main()
