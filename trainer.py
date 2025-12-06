"""Reusable training utilities for FSRS models."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt

from fsrs_optimizer import BatchDataset, BatchLoader  # type: ignore


def batch_process_wrapper(
    model: Any,
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    sequences, delta_ts, labels, seq_lens, weights = batch
    real_batch_size = seq_lens.shape[0]
    result = {"labels": labels, "weights": weights}
    outputs = model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
    result.update(outputs)
    return result


class Trainer:
    """Lightweight wrapper around BatchDataset/BatchLoader training loop."""

    def __init__(
        self,
        config,
        model: Any,
        train_set,
        test_set: Optional = None,
        batch_size: int = 256,
        max_seq_len: int = 64,
    ) -> None:
        self.config = config
        self.model = model.to(device=config.device)
        self.model.initialize_parameters(train_set)

        self.optimizer = self.model.get_optimizer(lr=self.model.lr, wd=self.model.wd)

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_epoch = self.model.n_epoch

        self.build_dataset(self.model.filter_training_data(train_set), test_set)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_data_loader.batch_nums * self.n_epoch
        )

        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set, test_set):
        self.train_set = BatchDataset(
            train_set.copy(),
            self.batch_size,
            max_seq_len=self.max_seq_len,
            device=self.config.device,
        )
        self.train_data_loader = BatchLoader(self.train_set)

        self.test_set = (
            []
            if test_set is None
            else BatchDataset(
                test_set.copy(),
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                device=self.config.device,
            )
        )
        self.test_data_loader = (
            [] if test_set is None else BatchLoader(self.test_set, shuffle=False)
        )

    def train(self):
        best_loss = float("inf")
        best_w = None
        epoch_len = len(self.train_set.y_train)

        for _ in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w

        for batch in self.train_data_loader:
                self.model.train()
                self.optimizer.zero_grad()
                result = batch_process_wrapper(self.model, batch, self.config.device)
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()
                self.model.apply_gradient_constraints()
                self.optimizer.step()
                self.scheduler.step()
                self.model.apply_parameter_clipper()

        weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w
        if best_w is None:
            raise RuntimeError("Training did not produce weights")
        return best_w, best_loss

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            self.train_data_loader.shuffle = False
            for data_loader in (self.train_data_loader, self.test_data_loader):
                if len(data_loader) == 0:
                    losses.append(0)
                    continue
                loss = 0
                total = 0
                epoch_len = len(data_loader.dataset.y_train)
                for batch in data_loader:
                    result = batch_process_wrapper(self.model, batch, self.config.device)
                    loss += (
                        (
                            self.loss_fn(result["retentions"], result["labels"])
                            * result["weights"]
                        )
                        .sum()
                        .detach()
                        .item()
                    )
                    if "penalty" in result:
                        loss += (result["penalty"] / epoch_len).detach().item()
                    total += batch[3].shape[0]
                losses.append(loss / total)
            self.train_data_loader.shuffle = True
            self.avg_train_losses.append(losses[0])
            self.avg_eval_losses.append(losses[1])

            w = self.model.state_dict()

            weighted_loss = (
                losses[0] * len(self.train_set) + losses[1] * len(self.test_set)
            ) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        self.avg_train_losses = [x.item() for x in self.avg_train_losses]
        self.avg_eval_losses = [x.item() for x in self.avg_eval_losses]
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig
