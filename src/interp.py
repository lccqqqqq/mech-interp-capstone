# Implement the linear moodule for hidden layers with sklearn
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
import torch as t
import numpy as np
import einops
from jaxtyping import Float, Int
from torch import Tensor
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from typing import Dict, Tuple
import re
import tsfm
import os


def parse_filename(filename, get_loss_data=False):
    pattern = r"^weights_lyr(\d+)_head(\d+)dh(\d+)_dm(\d+)_nsample(\d+)\.pth$"
    match = re.match(pattern, filename)
    if match:
        n_layers = int(match.group(1))
        n_heads = int(match.group(2))
        d_head = int(match.group(3))
        d_model = int(match.group(4))
        n_samples = int(match.group(5))

        arch_dict = {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_head": d_head,
            "d_model": d_model,
            "n_samples": n_samples,
        }

        if get_loss_data:
            # consider the corresponding file in loss
            loss_filename = filename.replace("weights", "losses")
            loss_dir = "C:/Users/15896/myenv/Hydra/mech_interp_capstone/data/Toylosses"
            loss_data = t.load(os.path.join(loss_dir, loss_filename))

            return arch_dict | loss_data

        return arch_dict

    raise ValueError("Invalid filename format")


def load_model(filename, device="cpu"):
    # get architecture from filename
    arch_params = parse_filename(filename)

    weights_dir = "C:/Users/15896/myenv/Hydra/mech_interp_capstone/data/Toyweights"

    data_dict = t.load(os.path.join(weights_dir, filename), map_location=device)

    model = tsfm.model_init(
        n_layers=arch_params["n_layers"],
        d_model=arch_params["d_model"],
        n_ctx=66,
        d_vocab=2,
        n_heads=arch_params["n_heads"],
        d_head=arch_params["d_head"],
        d_mlp=4 * arch_params["d_model"],
        device=device,
    )
    model.load_state_dict(data_dict)

    return model


def batch_load_model(directory, device="cpu"):
    weights_dir = directory
    filenames = os.listdir(weights_dir)
    models = {}
    for filename in filenames:
        model = load_model(filename, device=device)
        arch_params = parse_filename(filename)

        models[
            (arch_params["n_layers"], arch_params["d_model"], arch_params["n_heads"])
        ] = model

    return models


def format_resid_stream(
    cache: ActivationCache,
    model: HookedTransformer,
):
    stage_str = ["pre", "mid", "post"]
    resid = t.zeros(
        (
            3,  # stage: separated by attention block and MLP in each layer
            model.cfg.n_layers,  # layer
            cache["blocks.0.hook_resid_pre"].shape[0],  # get n_batch from cache
            model.cfg.n_ctx - 1,
            model.cfg.d_model,
        )
    )

    for lyr in range(model.cfg.n_layers):
        for stage in range(3):
            idx_str = f"blocks.{lyr}.hook_resid_{stage_str[stage]}"
            resid[stage, lyr, :, :, :] = cache[idx_str]

    return resid


def generate_targets(
    samples: Float[Tensor, "n_sample n_ctx d_model"],
    omegas: Float[Tensor, "n_sample"],
    times: Float,
    order=1,
    # TODO: Include the damping factor as well
):
    # canonical correlation analysis
    # encode the matrices
    # get left and right samples, for lead/lag in-context predictions
    samples_left = samples[:, :-1, :]
    samples_right = samples[:, 1:, :]

    A = t.zeros((samples_left.shape[0], samples_right.shape[1], 2, 2))
    rep_omega = einops.repeat(
        omegas, "n_sample -> n_sample n_ctx", n_ctx=samples_left.shape[1]
    )
    rep_dt = times[:, 1].unsqueeze(1).repeat(1, samples_left.shape[1])

    # print(A.shape, omegas.shape)
    A[:, :, 0, 0] = 0.0
    A[:, :, 0, 1] = rep_dt
    A[:, :, 1, 0] = -(rep_omega**2) * rep_dt
    A[:, :, 1, 1] = 0.0  # For zero damping

    # get targets by directly flattening the last two dimensions
    # NOTE: We are trying to map the hidden states to the target intermediate, which is itself an *operator*...

    A_propagate_dt = t.matrix_power(A, order)
    target = einops.rearrange(
        A_propagate_dt, "n_sample n_ctx i j -> n_sample n_ctx (i j)"
    )

    return target


def generate_periodicity_targets(
    samples: Float[Tensor, "n_sample n_ctx d_model"],
    omegas: Float[Tensor, "n_sample"],
    times: Float,
):
    A = t.zeros((samples.shape[0], samples.shape[1] - 1, 2))
    rep_omega = einops.repeat(
        omegas, "n_sample -> n_sample n_ctx", n_ctx=samples.shape[1] - 1
    )
    rep_dt = times[:, 1].unsqueeze(1).repeat(1, samples.shape[1] - 1)

    rep_phase = rep_omega * rep_dt
    A[:, :, 0] = t.cos(rep_phase)
    A[:, :, 1] = t.sin(rep_phase)

    target = A  # (stupid practise but we don't need to further process this)

    return target


def generate_matexp_targets(
    samples: Float[Tensor, "n_sample n_ctx d_model"],
    omegas: Float[Tensor, "n_sample"],
    times: Float,
):
    A = t.zeros((samples.shape[0], samples.shape[1] - 1, 2, 2))
    rep_omega = einops.repeat(
        omegas, "n_sample -> n_sample n_ctx", n_ctx=samples.shape[1] - 1
    )
    rep_dt = times[:, 1].unsqueeze(1).repeat(1, samples.shape[1] - 1)

    rep_phase = rep_omega * rep_dt
    A[:, :, 0, 0] = t.cos(rep_phase)
    A[:, :, 0, 1] = t.sin(rep_phase) / rep_omega
    A[:, :, 1, 0] = -rep_omega * t.sin(rep_phase)
    A[:, :, 1, 1] = t.cos(rep_phase)

    target = einops.rearrange(A, "n_sample n_ctx i j -> n_sample n_ctx (i j)")

    return target


def lin_reg(
    input: Float[Tensor, "n_sample n_ctx d_model"],
    target: Float[Tensor, "n_sample n_ctx targ"],
    reverse=False,  # if true, use linreg against target -> input (hidden layer states)
    trunc=None,  # or int: number of samples the regression is performed on
    alpha=1.0,
    seed=None,
):
    if seed is not None:
        t.manual_seed(seed)
        np.random.seed(seed)

    np_input = input[:, 1:, :].detach().cpu().numpy()
    np_target = target.detach().cpu().numpy()

    if trunc is not None:
        id_row = np.random.choice(np_input.shape[0], trunc, replace=False)
        np_input = np_input[id_row]
        np_target = np_target[id_row]

    # reshape
    np_input = einops.rearrange(
        np_input, "n_sample n_ctx d_model -> (n_sample n_ctx) d_model"
    )
    np_target = einops.rearrange(
        np_target, "n_sample n_ctx targ -> (n_sample n_ctx) targ"
    )

    clf = Ridge(alpha=alpha)
    if not reverse:
        clf.fit(np_input, np_target)
        r2 = clf.score(np_input, np_target)
        mse = np.mean((clf.predict(np_input) - np_target) ** 2)
    else:
        clf.fit(np_target, np_input)
        r2 = clf.score(np_target, np_input)
        mse = np.mean((clf.predict(np_target) - np_input) ** 2)

    return r2, mse


def get_annotations(cache: ActivationCache):
    hook_str = cache.keys()
    annotation_table = {}

    for hook in hook_str:
        layer = re.findall(r"\d+", hook)
        if layer != []:
            layer = layer[0]

        if "mlp" in hook:
            if "post" in hook:
                annotation_table[hook] = "post-mlp" + layer
            elif "pre" in hook:
                annotation_table[hook] = "pre-mlp" + layer
        else:
            if "resid" in hook:
                if "post" in hook:
                    annotation_table[hook] = "resid-post" + layer
                elif "pre" in hook:
                    annotation_table[hook] = "resid-pre" + layer
                elif "mid" in hook:
                    annotation_table[hook] = "resid-mid" + layer

    # annotation_table = {v: k for k, v in annotation_table.items()}

    return annotation_table


def maxR2_models(
    model_list: Dict[Tuple[Int, Int, Int], HookedTransformer],
    target: Float[Tensor, "n_sample n_ctx targ"],
    samples: Float[Tensor, "n_sample n_ctx d_vocab"],
    reverse=False,
    trunc=1000,
    alpha=1.0,
):

    # For given model, calculate the maximum R2 score across all layers
    max_r2_dict = {}
    for model_key, model in model_list.items():
        pred, cache = model.run_with_cache(samples)
        max_r2 = 0
        max_r2_pos = None
        anno_tab = get_annotations(cache)

        for elem in anno_tab.keys():
            input = cache[elem]
            r2, mse = lin_reg(input, target, reverse=reverse, trunc=trunc, alpha=alpha)
            if r2 > max_r2:
                max_r2 = r2
                max_r2_pos = anno_tab[elem]

        max_r2_dict[model_key] = (max_r2, max_r2_pos)

    return max_r2_dict


def generate_heatmap(max_r2_dict: Dict[Tuple[Int, Int, Int], Tuple[Float, str]]):
    # The key inds: (layers, d_model, heads)
    # The value inds: (max_r2, position)
    # x-axis: layers
    # y-axis: d_model
    # color: r2 score
    # generate separate heatmaps for each head count

    # get distinct elements
    n_layers = set(sorted([key_tup[0] for key_tup in max_r2_dict.keys()]))
    d_models = set(sorted([key_tup[1] for key_tup in max_r2_dict.keys()]))
    n_heads = set(sorted([key_tup[2] for key_tup in max_r2_dict.keys()]))

    # create index sets
    lyr_indmap = {lyr: i for i, lyr in enumerate(n_layers)}
    dm_indmap = {dm: i for i, dm in enumerate(d_models)}
    nh_indmap = {nh: i for i, nh in enumerate(n_heads)}

    data = [np.zeros((len(n_layers), len(d_models))) for _ in range(len(n_heads))]
    annotation = [
        np.empty((len(n_layers), len(d_models)), dtype=str) for _ in range(len(n_heads))
    ]
    annotation = [
        [["" for _ in range(len(d_models))] for _ in range(len(n_layers))]
        for _ in range(len(n_heads))
    ]
    # ylabels = [f"nh-{nh}" for nh in n_heads]
    # xlabels = [f"lyr-{lyr}" for lyr in n_layers]

    for key, val in max_r2_dict.items():
        # unpack key and values
        lyr, dm, nh = key
        r2, pos = val

        data[nh_indmap[nh]][lyr_indmap[lyr], dm_indmap[dm]] = r2
        annotation[nh_indmap[nh]][lyr_indmap[lyr]][dm_indmap[dm]] = pos + f"\n {r2:.3f}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    print(annotation)

    # get keys

    head_count = len(set(n_heads))

    for i in range(head_count):
        # get data matrix
        # plot
        sns.heatmap(
            data=data[i],
            annot=annotation[i],
            fmt="",
            cmap="viridis",
            cbar=True,
            ax=axes[i],
            xticklabels=list(d_models),
            yticklabels=list(n_layers),
        )

    plt.show()
    plt.xlabel("d_model")
    plt.ylabel("n_layers")


def data_generator(
    num_samples: Int,
    seq_len: Int,
    device="cpu",
    seed=None,
):
    if seed is not None:
        t.manual_seed(seed)
        np.random.seed(seed)

    train_data, train_w, test_data, test_w, train_t, test_t = tsfm.generate_springdata(
        num_samples=num_samples, sequence_length=seq_len, device=device
    )

    samples = t.cat([train_data, test_data], dim=0)
    omegas = t.cat([train_w, test_w], dim=0)
    times = t.cat([train_t, test_t], dim=0)

    # randomly shuffle
    idx = t.randperm(samples.size(0))
    samples = samples[idx]
    omegas = omegas[idx]
    times = times[idx]

    return samples, omegas, times


def interp_with_ablation(
    model: HookedTransformer,
    window: Int,
    samples: Float[Tensor, "n_sample n_ctx d_vocab"],
):
    model.eval()

    # original loss
    pred = model(samples)
    loss_baseline = t.nn.functional.mse_loss(pred[:, :-1, :], samples[:, 1:, :])

    abl_losses = []
    for start in range(0, 64 - window - 1):
        # ablation
        samples_abl = samples.clone()
        samples_abl[:, start : start + window, :] = 0.0
        pred_abl = model(samples_abl)

        # also ablated the prediction to zero out the loss over the ablated datapoints as we should not expect the model to predict where we ablate the data
        pred_abl[:, start - 1 : start + window - 1, :] = pred[
            :, start - 1 : start + window - 1, :
        ]
        loss_abl = t.nn.functional.mse_loss(pred_abl[:, :-1, :], samples_abl[:, 1:, :])
        abl_losses.append(loss_abl)

    abl_losses = t.stack(abl_losses)

    return abl_losses, loss_baseline


def interp_with_noise(
    model: HookedTransformer,
    noise_level: Float,
    samples: Float[Tensor, "n_sample n_ctx d_vocab"],
    seed=None,
):
    if seed is not None:
        t.manual_seed(seed)
        np.random.seed(seed)

    model.eval()

    # original loss
    pred = model(samples)
    loss_baseline = t.nn.functional.mse_loss(pred[:, :-1, :], samples[:, 1:, :])

    # add noise
    samples_noisy = samples + noise_level * t.randn_like(samples)
    pred_noisy = model(samples_noisy)
    loss_noisy = t.nn.functional.mse_loss(
        pred_noisy[:, :-1, :], samples_noisy[:, 1:, :]
    )

    return loss_noisy, loss_baseline


if __name__ == "__main__":
    dir_name = "C:/Users/15896/myenv/Hydra/mech_interp_capstone/data/Toyweights"
    models = batch_load_model(dir_name)

    samples, omegas, times = data_generator(200, 64)
    # targets = generate_matexp_targets(samples, omegas, times)
    # max_r2_dict = maxR2_models(
    #     models, targets, samples, reverse=True, trunc=100, alpha=3.0
    # )

    # generate_heatmap(max_r2_dict)

    windows = [1, 3, 5, 10]
    window = windows[1]
    for model in models.values():
        # filter the model
        if model.cfg.n_heads == 2 and model.cfg.n_layers == 2:
            abl_losses, loss_baseline = interp_with_ablation(model, window, samples)
            plt.plot(abl_losses.detach().numpy(), label=f"model: d={model.cfg.d_model}")
            plt.axhline(
                loss_baseline.detach().numpy(),
                color=plt.gca().lines[-1].get_color(),
                linestyle="--",
                linewidth=0.5,
                label="baseline",
            )

    plt.legend()
    plt.xlabel("Ablation Window start position")
    plt.ylabel("MSE Ablated Loss")
    plt.show()
