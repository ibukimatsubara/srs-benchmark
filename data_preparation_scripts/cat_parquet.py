#!/usr/bin/env python3
"""Prints the content of a parquet file to stdout."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILE = "test_maimemo_parquet_cefr/revlogs/user_id=9999/data.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print (cat) the rows contained in a parquet file"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=DEFAULT_FILE,
        help="Path to the parquet file (defaults to the requested test file).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit to the number of rows printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.file)

    if not parquet_path.exists():
        raise SystemExit(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(df.to_string())


if __name__ == "__main__":
    main()
