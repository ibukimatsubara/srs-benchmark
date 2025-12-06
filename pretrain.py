from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm  # type: ignore

from config import Config, create_parser
from data_loader import UserDataLoader
from models.model_factory import create_model
from trainer import Trainer


CHECKPOINT_INTERVAL = 1000


def train_and_save(config: Config, dataset: pd.DataFrame, path: Path) -> dict:
    model = create_model(config)
    trainer = Trainer(
        config,
        model=model,
        train_set=dataset,
        test_set=None,
        batch_size=config.batch_size,
    )
    weights = trainer.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, path)
    print(f"Saved pretrained weights to {path}")
    return weights


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--output", required=True, help="Path to save pretrained weights")
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Limit the number of users loaded for pretraining",
    )
    args = parser.parse_args()
    config = Config(args)

    if config.model_name not in {"FSRS-6", "FSRS-6-cefr"}:
        raise SystemExit("pretrain.py currently supports only FSRS-6 variants.")

    revlog_root = config.data_path / "revlogs"
    if not revlog_root.exists():
        raise SystemExit(f"Revlog directory not found: {revlog_root}")

    user_dirs = sorted(revlog_root.glob("user_id=*"))
    if args.max_users is not None:
        user_dirs = user_dirs[: args.max_users]

    if not user_dirs:
        raise SystemExit("No user directories found for pretraining.")
    loader = UserDataLoader(config)
    frames: list[pd.DataFrame] = []
    output_path = Path(args.output)
    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_users = 0
    last_checkpoint_weights: Optional[dict] = None

    for idx, user_dir in enumerate(tqdm(user_dirs, desc="Loading users"), start=1):
        user_id = user_dir.name.replace("user_id=", "")
        df = loader.load_user_data(user_id)
        frames.append(df)

        if idx % CHECKPOINT_INTERVAL == 0:
            dataset = pd.concat(frames, ignore_index=True)
            ckpt_path = checkpoint_dir / f"{output_path.stem}_users{idx}.pth"
            last_checkpoint_weights = train_and_save(config, dataset, ckpt_path)
            last_checkpoint_users = idx

    total_users = len(frames)
    full_dataset = pd.concat(frames, ignore_index=True)

    if total_users == last_checkpoint_users and last_checkpoint_weights is not None:
        torch.save(last_checkpoint_weights, output_path)
        print(f"Saved pretrained weights to {output_path}")
    else:
        train_and_save(config, full_dataset, output_path)
