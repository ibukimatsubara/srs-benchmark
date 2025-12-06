"""Print selected parameters from a saved FSRS state dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from typing import Any, Mapping

BASELINE_FSRS6 = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
]

BASELINE_FSRS6_CEFR = BASELINE_FSRS6 + [3.0, 4.0, 5.0, 6.5, 8.0, 9.0]

BASELINE_MAP = {len(BASELINE_FSRS6): BASELINE_FSRS6, len(BASELINE_FSRS6_CEFR): BASELINE_FSRS6_CEFR}

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


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

    baseline = BASELINE_MAP.get(len(weights))

    print(f"Total parameters: {len(weights)}")

    def format_value(idx: int, value: float) -> str:
        if not baseline:
            return f"{value:.6f}"
        delta = value - baseline[idx]
        color = GREEN if delta > 1e-9 else RED if delta < -1e-9 else ""
        delta_text = f"{delta:+.6f}"
        base_text = f"{baseline[idx]:.6f}"
        if color:
            delta_text = f"{color}{delta_text}{RESET}"
        return f"{value:.6f} ({delta_text} vs {base_text})"

    if args.cefr_only:
        print("CEFR difficulty parameters (w[21-26]):")
        for offset, level in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"], start=21):
            if offset >= len(weights):
                break
            value = weights[offset]
            formatted = format_value(offset, value)
            print(f"  {level}: {formatted}")
    else:
        for idx, value in enumerate(weights):
            formatted = format_value(idx, value)
            print(f"w[{idx:02d}] = {formatted}")


if __name__ == "__main__":
    main()
