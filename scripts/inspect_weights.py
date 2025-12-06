"""Print selected parameters from a saved FSRS state dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from typing import Any, Mapping


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

    def extract_weights(obj: Any):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)) and all(
            isinstance(x, (int, float)) for x in obj
        ):
            return torch.tensor(obj)
        if isinstance(obj, Mapping):
            if "w" in obj:
                return torch.as_tensor(obj["w"])
            for key in ("model_state_dict", "state_dict"):
                if key in obj and isinstance(obj[key], Mapping) and "w" in obj[key]:
                    return torch.as_tensor(obj[key]["w"])
            for value in obj.values():
                result = extract_weights(value)
                if result is not None and result.ndim == 1:
                    return result
        if isinstance(obj, (list, tuple)):
            for item in obj:
                result = extract_weights(item)
                if result is not None:
                    return result
        return None

    weights_tensor = extract_weights(state)
    if weights_tensor is None:
        raise SystemExit("Could not locate parameter vector 'w' in the file")

    weights = weights_tensor.tolist()

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
