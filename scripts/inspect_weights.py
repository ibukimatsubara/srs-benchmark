"""Print selected parameters from a saved FSRS state dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FSRS weight file")
    parser.add_argument("path", help="Path to .pth state dict")
    parser.add_argument(
        "--cefr-only",
        action="store_true",
        help="Only print CEFR difficulty parameters (w[21:27])",
    )
    args = parser.parse_args()

    pth = Path(args.path)
    if not pth.exists():
        raise SystemExit(f"File not found: {pth}")

    state = torch.load(pth, map_location="cpu")

    if isinstance(state, dict) and "w" in state:
        weights = state["w"]
    else:
        raise SystemExit("State dict does not contain key 'w'")

    weights = torch.as_tensor(weights).tolist()

    if args.cefr_only:
        print("CEFR difficulty parameters (w[21-26]):")
        for level, value in zip(["A1", "A2", "B1", "B2", "C1", "C2"], weights[21:27]):
            print(f"  {level}: {value:.6f}")
    else:
        print(f"Total parameters: {len(weights)}")
        for idx, value in enumerate(weights):
            print(f"w[{idx:02d}] = {value:.6f}")


if __name__ == "__main__":
    main()
