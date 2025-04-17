# Description: This file contains the necessary imports for the transformer_lens package and other necessary packages.

import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import os

# import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint
import math
from torch.utils.data import TensorDataset, DataLoader

# for running multiple params
import argparse


print("CUDA available:", t.cuda.is_available())
t.manual_seed(42)
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda" if t.cuda.is_available() else "cpu"
)


### Data generation


def get_random_start(target_size):
    # can be used to get x0, v0
    random_numbers = 2 * t.rand(target_size[0]) - 1
    # Expand each number to a row of 65 identical elements
    expanded_tensor = random_numbers.unsqueeze(1).expand(-1, target_size[1])
    return expanded_tensor


def generate_springdata(
    num_samples=1000, sequence_length=10, device=device, save_data=True
):
    # Generate x = cos(wt), v = -w*sin(wt). trains on omegas between 0.5pi and 1pi, tests on 0.25pi-0.5pi and 1pi-1.25pi

    omegas_range = [0.25 * t.pi, 1.25 * t.pi]
    delta_omega = omegas_range[1] - omegas_range[0]

    train_omegas = (
        t.rand(num_samples) * delta_omega / 2 + omegas_range[0] + delta_omega / 4
    )
    # middle half of the omega interval is the training set
    train_deltat = (
        t.rand(num_samples) * 2 * t.pi / (train_omegas)
    )  # cos(wt) has period 2pi/w. so deltat>2pi/w is redundant

    start = 0
    skip = 1
    train_times = (
        t.arange(start, start + skip * sequence_length + 1, step=skip)
        .unsqueeze(0)
        .repeat(num_samples, 1)
    )
    train_times = train_times * train_deltat.unsqueeze(1)
    train_omegas_unsq = train_omegas.unsqueeze(1)

    x0_train, v0_train = get_random_start(train_times.shape), get_random_start(
        train_times.shape
    )
    x_train = x0_train * t.cos(train_omegas_unsq * train_times) + (
        v0_train / train_omegas_unsq
    ) * t.sin(train_omegas_unsq * train_times)
    v_train = -x0_train * train_omegas_unsq * t.sin(
        train_omegas_unsq * train_times
    ) + v0_train * t.cos(train_omegas_unsq * train_times)
    # stack x and v
    sequences_train = t.stack(
        (x_train, v_train), dim=2
    )  # Shape: (num_samples, sequence_length, 2)

    test_omegas_low = t.rand(num_samples // 4) * delta_omega / 4 + omegas_range[0]
    test_omegas_high = (
        t.rand(num_samples // 4) * delta_omega / 4 + omegas_range[1] - delta_omega / 4
    )

    # concatenate the two
    test_omegas = t.cat((test_omegas_low, test_omegas_high))
    test_deltat = t.rand(test_omegas.shape[-1]) * 2 * t.pi / test_omegas

    test_times = (
        t.arange(start, start + skip * sequence_length + 1, step=skip)
        .unsqueeze(0)
        .repeat(test_deltat.shape[-1], 1)
    )
    test_times = test_times * test_deltat.unsqueeze(1)
    test_omegas_unsq = test_omegas.unsqueeze(1)

    x0_test, v0_test = get_random_start(test_times.shape), get_random_start(
        test_times.shape
    )
    x_test = x0_test * t.cos(test_omegas_unsq * test_times) + (
        v0_test / test_omegas_unsq
    ) * t.sin(test_omegas_unsq * test_times)
    v_test = -x0_test * test_omegas_unsq * t.sin(
        test_omegas_unsq * test_times
    ) + v0_test * t.cos(test_omegas_unsq * test_times)
    # stack x and v
    sequences_test = t.stack(
        (x_test, v_test), dim=2
    )  # Shape: (num_samples, sequence_length, 2)

    # Create directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save spring_data
    t.save(
        {
            "sequences_train": sequences_train,
            "train_omegas": train_omegas,
            "sequences_test": sequences_test,
            "test_omegas": test_omegas,
            "train_times": train_times,
            "test_times": test_times,
        },
        f"data/undamped.pth",
    )

    return (
        sequences_train.to(device),
        train_omegas,
        sequences_test.to(device),
        test_omegas,
        train_times,
        test_times,
    )


### The model: Transformer with modified embedding layer


class PhaseSpaceEmbed(nn.Module):
    def __init__(self, d_model: int, d_phase_space: int):
        super().__init__()
        self.d_model = d_model
        self.d_phase_space = d_phase_space
        self.W_E = nn.Parameter(t.randn(d_model, d_phase_space))
        self.b_E = nn.Parameter(t.randn(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.W_E, self.b_E)


class TimeEmbed(nn.Module):
    def __init__(self, seq_len: int, d_model: int, *kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.W_pos = nn.Parameter(t.zeros(seq_len, d_model))
        # initialize to zero embedding

    def forward(self, x: Tensor, *kwargs) -> Tensor:
        seq_len = x.shape[1]
        positions = t.arange(0, seq_len, dtype=t.long, device=x.device)
        embedded_pos = self.W_pos[positions]

        return embedded_pos


def model_init(
    n_layers: int,
    d_model: int,
    n_ctx: int,
    d_vocab: int,
    n_heads: int,
    d_head: int,
    d_mlp: int,
    device=device,
    attn_only=False,
):
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        n_heads=n_heads,
        d_mlp=d_mlp,
        act_fn="gelu",  # set to defaule gaussain error linear unit
        normalization_type=None,
        attn_only=attn_only,
    )

    model = HookedTransformer(cfg)
    model.embed = PhaseSpaceEmbed(cfg.d_model, cfg.d_vocab)
    model.pos_embed = TimeEmbed(cfg.n_ctx, cfg.d_model)
    model.to(device)

    return model


def model_train_init(
    n_layers: int,
    d_model: int,
    n_ctx: int,
    d_vocab: int,
    n_heads: int,
    d_head: int,
    d_mlp: int,
    num_samples=1000,
    sequence_length=50,
    lr=1e-3,
    batch_size=64,
    n_epochs=200,
    checkpoint_every=10,
    betas=(0.9, 0.98),
    device=device,
    attn_only=False,
):
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        n_heads=n_heads,
        d_mlp=d_mlp,
        act_fn="gelu",  # set to defaule gaussain error linear unit
        normalization_type=None,
        attn_only=attn_only,
    )
    model = HookedTransformer(cfg)
    model.embed = PhaseSpaceEmbed(cfg.d_model, cfg.d_vocab)
    model.pos_embed = TimeEmbed(cfg.n_ctx, cfg.d_model)
    model.to(device)

    train_cfg = {
        "lr": lr,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "checkpoint_every": checkpoint_every,
        "betas": betas,
    }

    train_data, _, test_data, _, _, _ = generate_springdata(
        num_samples, sequence_length, device=device
    )

    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg["batch_size"], shuffle=False
    )

    return model, train_cfg, train_loader, test_loader, train_data, test_data


def loop(
    model,
    train_loader,
    train_data,
    test_data,
    train_cfg,
    save_weights=True,
    save_loss=True,
):

    optimizer = t.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], betas=train_cfg["betas"]
    )

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(train_cfg["n_epochs"])):
        model.train()
        for (minibatch,) in train_loader:
            minibatch_prediction = model(minibatch)
            train_loss = F.mse_loss(
                minibatch_prediction[:, :-1, :], minibatch[:, 1:, :]
            )
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation phase
        if epoch % train_cfg["checkpoint_every"] == 0:
            model.eval()
            with t.inference_mode():
                # Evaluate on all training data
                full_train_pred = model(train_data)
                train_loss = F.mse_loss(
                    full_train_pred[:, :-1, :], train_data[:, 1:, :]
                )

                # Evaluate on test data
                test_pred = model(test_data)
                test_loss = F.mse_loss(test_pred[:, :-1, :], test_data[:, 1:, :])

                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())

                print(
                    f"Epoch {epoch}: Train loss {train_loss.item()}, Test loss {test_loss.item()}"
                )

    # save the model
    os.makedirs("Toyweights", exist_ok=True)
    os.makedirs("Toylosses", exist_ok=True)
    if save_weights:
        t.save(
            model.state_dict(),
            f"Toyweights/weights_lyr{model.cfg.n_layers}_head{model.cfg.n_heads}dh{model.cfg.d_head}_dm{model.cfg.d_model}_nsample{train_cfg['n_epochs']}.pth",
        )

    if save_loss:
        t.save(
            {"train_losses": train_losses, "test_losses": test_losses},
            f"Toylosses/losses_lyr{model.cfg.n_layers}_head{model.cfg.n_heads}dh{model.cfg.d_head}_dm{model.cfg.d_model}_nsample{train_cfg['n_epochs']}.pth",
        )

    print("Training complete!")

    pass


############################################

# running the attention-only model

if __name__ == "__main__":
    print("Generating spring data")

    ## running multiple arguments passed from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_mlp", type=int, default=4 * 16)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_head", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--checkpoint_every", type=int, default=10)

    args = parser.parse_args()

    model, train_cfg, train_loader, test_loader, train_data, test_data = (
        model_train_init(
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_ctx=66,
            d_vocab=2,
            n_heads=args.n_heads,
            d_head=args.d_head,
            # d_mlp=args.d_mlp,
            num_samples=args.num_samples,
            sequence_length=50,
            lr=1e-3,
            batch_size=64,
            n_epochs=args.n_epochs,
            checkpoint_every=args.checkpoint_every,
            betas=(0.9, 0.98),
            device=device,
            attn_only=True,
        )
    )

    loop(
        model,
        train_loader,
        train_data,
        test_data,
        train_cfg,
        save_weights=True,
        save_loss=True,
    )


############################################

# running the model

# if __name__ == "__main__":
#     print("Generating spring data")

#     ## running multiple arguments passed from the command line
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_layers", type=int, default=2)
#     parser.add_argument("--d_model", type=int, default=16)
#     parser.add_argument("--d_mlp", type=int, default=4 * 16)
#     parser.add_argument("--n_heads", type=int, default=4)
#     parser.add_argument("--d_head", type=int, default=4)
#     parser.add_argument("--num_samples", type=int, default=1000)
#     parser.add_argument("--n_epochs", type=int, default=20)
#     parser.add_argument("--checkpoint_every", type=int, default=10)

#     args = parser.parse_args()

#     model, train_cfg, train_loader, test_loader, train_data, test_data = (
#         model_train_init(
#             n_layers=args.n_layers,
#             d_model=args.d_model,
#             n_ctx=66,
#             d_vocab=2,
#             n_heads=args.n_heads,
#             d_head=args.d_head,
#             d_mlp=args.d_mlp,
#             num_samples=args.num_samples,
#             sequence_length=50,
#             lr=1e-3,
#             batch_size=64,
#             n_epochs=args.n_epochs,
#             checkpoint_every=args.checkpoint_every,
#             betas=(0.9, 0.98),
#             device=device,
#         )
#     )

#     loop(
#         model,
#         train_loader,
#         train_data,
#         test_data,
#         train_cfg,
#         save_weights=True,
#         save_loss=True,
#     )
