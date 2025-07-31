# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# %%
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
from argparse import ArgumentParser

torch.set_float32_matmul_precision("high")

from grooming_graph.operators import RandMode, OperatorChain
from grooming_graph.strands import Strands, StrandParameterization
from grooming_graph.operators.guides import compute_guide_from_root_and_dirs
from head import HeadModel
from renderer import Renderer, render_grooms
from scene import Scene
from dataset_readers import (
    readSyntheticCameras,
    readMonoHairCameras,
    readGaussianHaircutCameras,
)
from optimization.guide_initialization import get_visible_strand_idx, initialize_guides
from optimization.instancing import optimize_instancing
from optimization.guide_optimization import optimize_guides
from optimization.operator_parameter import optimize_operator_parameters
from optimization.operator_random import optimize_operator_random
from optimization.reassign_operator_guides import reassign_operator_guides

torch.manual_seed(0)
np.random.seed(0)

# -------------------------------------------------------------------------------------------------
# Config and output directory setup
# -------------------------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--dont_save_results", action="store_true", default=False)
parser.add_argument("--use_known_guides", action="store_true", default=False)
parser.add_argument("--use_known_params", action="store_true", default=False)
parser.add_argument("--no_guide_opt", action="store_true", default=False)
parser.add_argument("--keep_invisible_strands", action="store_true", default=False)
parser.add_argument("--use_same_rng", action="store_true", default=False)
args, _ = parser.parse_known_args(sys.argv[1:])

dataset = args.dataset or "synthetic-grooms-coily"
dataset_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"data/{dataset}")
)
is_synthetic = "synthetic" in dataset_path
dataset_name = os.path.basename(dataset_path)

print("Running dataset:", dataset_name)

save_results = not args.dont_save_results
USE_KNOWN_GUIDES = args.use_known_guides and is_synthetic
USE_KNOWN_PARAMS = args.use_known_params and is_synthetic
NO_GUIDE_OPT = args.no_guide_opt
INTERPOLATE_INVISIBLE_STRANDS = not args.keep_invisible_strands and not is_synthetic
SAME_RNG = args.use_same_rng
RAND_MODE = RandMode.OPTIMIZE if not SAME_RNG else RandMode.FIX

DEFAULT_SMOOTH_STEPS = 80
INSTANCING_LR = 1e-2
INSTANCING_N_ITERATIONS = 2000
GUIDES_CURV_LR = 5e-2
GUIDES_LENGTH_LR = 1.5e-3
GUIDES_ROOT_DIR_LR = 5e-2
GUIDES_N_ITERATIONS = 3000
SMOOTHING_LAMBDA = 10.0
SIGMA_D = 0.3
OPERATOR_LR = 1e-2
OPERATOR_N_ITERATIONS = 3000
OPERATOR_RAND_LR = 5e-2
OPERATOR_RAND_N_ITERATIONS = 3000
OPERATOR_RAND_NOISE_FACTOR_BEGIN = 5.0
OPERATOR_RAND_NOISE_FACTOR_END = 0.0
REASSIGN_GUIDES_K = 16
REASSIGN_GUIDES_THRESHOLD = 0.2

config = {
    "USE_KNOWN_GUIDES": USE_KNOWN_GUIDES,
    "USE_KNOWN_PARAMS": USE_KNOWN_PARAMS,
    "NO_GUIDE_OPT": NO_GUIDE_OPT,
    "INTERPOLATE_INVISIBLE_STRANDS": INTERPOLATE_INVISIBLE_STRANDS,
    "SAME_RNG": SAME_RNG,
    "RAND_MODE": int(RAND_MODE),
    "DEFAULT_SMOOTH_STEPS": DEFAULT_SMOOTH_STEPS,
    "INSTANCING_LR": INSTANCING_LR,
    "INSTANCING_N_ITERATIONS": INSTANCING_N_ITERATIONS,
    "GUIDES_CURV_LR": GUIDES_CURV_LR,
    "GUIDES_LENGTH_LR": GUIDES_LENGTH_LR,
    "GUIDES_ROOT_DIR_LR": GUIDES_ROOT_DIR_LR,
    "GUIDES_N_ITERATIONS": GUIDES_N_ITERATIONS,
    "SMOOTHING_LAMBDA": SMOOTHING_LAMBDA,
    "SIGMA_D": SIGMA_D,
    "OPERATOR_LR": OPERATOR_LR,
    "OPERATOR_N_ITERATIONS": OPERATOR_N_ITERATIONS,
    "OPERATOR_RAND_LR": OPERATOR_RAND_LR,
    "OPERATOR_RAND_N_ITERATIONS": OPERATOR_RAND_N_ITERATIONS,
    "OPERATOR_RAND_NOISE_FACTOR_BEGIN": OPERATOR_RAND_NOISE_FACTOR_BEGIN,
    "OPERATOR_RAND_NOISE_FACTOR_END": OPERATOR_RAND_NOISE_FACTOR_END,
    "REASSIGN_GUIDES_K": REASSIGN_GUIDES_K,
    "REASSIGN_GUIDES_THRESHOLD": REASSIGN_GUIDES_THRESHOLD,
}

if save_results:
    out_dir = os.path.join("output", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    last_it = (
        sorted([int(d) for d in os.listdir(out_dir)])[-1] if os.listdir(out_dir) else 0
    )
    out_dir = os.path.join(out_dir, str(last_it + 1))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Saving results to {out_dir}")

# -------------------------------------------------------------------------------------------------
# Load scene
# -------------------------------------------------------------------------------------------------

scene_file_path = os.path.join(dataset_path, "scene-3d.py")
print(f"Reading scene file from {scene_file_path}...")
scene_dict = {"__file__": scene_file_path}
with open(scene_file_path) as f:
    exec(f.read(), scene_dict)

if save_results:
    # copy scene file to output directory
    with open(os.path.join(out_dir, "scene-3d.py"), "w") as f:
        with open(scene_file_path) as scene_f:
            f.write(scene_f.read())

    # copy run file to output directory
    with open(os.path.join(out_dir, "train-3d.py"), "w") as f:
        with open(__file__) as run_f:
            f.write(run_f.read())

print("Loading head model...")
head = HeadModel(dataset_path, scene_dict.get("transform", None))

print("Loading groom and guide data...")
n_guides, target_guides, target_groom = scene_dict["load_groom_guides"](head)

roots = target_groom[:, 0].clone()
if is_synthetic:
    roots = head.resample_roots(roots, int(len(roots) * 1.25))

n_strands = roots.shape[0]

print("Loading cameras...")
renderer = Renderer(head=head)

if os.path.exists(os.path.join(dataset_path, "transforms_train.json")):
    train_cameras = readSyntheticCameras(dataset_path, "transforms_train.json")
elif os.path.exists(os.path.join(dataset_path, "cam_params.json")):
    train_cameras = readMonoHairCameras(dataset_path, "cam_params.json")
elif os.path.exists(os.path.join(dataset_path, "cameras.pkl")):
    train_cameras = readGaussianHaircutCameras(dataset_path, "cameras.pkl")


def render(*grooms, save_name=None):
    view = scene_dict.get("default_view", 21)

    render_grooms(
        renderer,
        train_cameras[view],
        *grooms,
        save=save_results,
        save_path=(
            os.path.join(out_dir, save_name + ".png")
            if save_results and save_name
            else None
        ),
    )


# Initialize grooming chains
print("Initializing grooming chain...")
grooming_chain: OperatorChain = scene_dict["initialize_grooming_chain"](
    n_strands,
    n_guides,
    head=head,
    seed=0 if SAME_RNG else 3,
    rand_mode=RAND_MODE,
)
grooming_chain.to(target_groom.device)

if is_synthetic:
    # To validate synthetic examples, make sure target is generated using the same procedural
    # pipeline with the same operators but with different parameters
    grooming_chain_target: OperatorChain = scene_dict[
        "initialize_target_grooming_chain"
    ](n_strands, target_guides.shape[0], head=head, seed=0, rand_mode=RAND_MODE)
    grooming_chain_target.to(target_groom.device)

    target_groom = grooming_chain_target(
        roots, target_guides, head.scalp_get_normal(target_guides[:, 0])
    ).detach()

# Run once using dummy guides to make chain and target chain random sequence consistent if needed
guide_roots = head.scalp_inverse_uv_mapping(head.scalp_sample_uvs(n_guides))
guide_normals = head.scalp_get_normal(guide_roots).unsqueeze(1)
initial_guides = compute_guide_from_root_and_dirs(guide_roots, guide_normals)
grooming_chain(roots, initial_guides, guide_normals[:, 0])

if USE_KNOWN_PARAMS:
    with torch.no_grad():
        for op, op_target in zip(
            grooming_chain.operators, grooming_chain_target.operators
        ):
            for p, p_target in zip(op.parameters(), op_target.parameters()):
                if not p.is_random():
                    p[...] = p_target

render(
    {
        "strands": target_groom,
        "title": "Reference",
    },
    {
        "strands": target_guides,
        "title": "Reference guides",
    },
    save_name="reference",
)

scene = Scene(
    head=head,
    renderer=renderer,
    render_fn=render,
    train_cameras=train_cameras,
    is_synthetic=is_synthetic,
    grooming_chain=grooming_chain,
    target_groom=target_groom,
    target_guides=target_guides,
    opt_strand_idx=None,
    roots=roots,
    cluster_guides=None,
    initial_guides=None,
    guide_roots=None,
    guide_strands=None,
)

# %%
# -------------------------------------------------------------------------------------------------
# Stage 1: Guide Initialization
# -------------------------------------------------------------------------------------------------
print("Stage 1: Initializing guides...")
if INTERPOLATE_INVISIBLE_STRANDS:
    opt_strand_idx = torch.from_numpy(get_visible_strand_idx(dataset_path, scene)).to(
        target_groom.device
    )
    target_groom = target_groom[opt_strand_idx]
else:
    opt_strand_idx = torch.arange(n_strands, device=target_groom.device)
scene.target_groom = target_groom
scene.opt_strand_idx = opt_strand_idx

if USE_KNOWN_GUIDES:
    initial_guides = target_guides.clone()
    cluster_guides = target_guides.clone()
    scene.initial_guides = initial_guides
    scene.cluster_guides = cluster_guides
else:
    cluster_operator_guides = not is_synthetic
    initial_guides, cluster_guides = initialize_guides(
        scene,
        n_guides,
        scene_dict.get("smooth_steps", DEFAULT_SMOOTH_STEPS),
        cluster_operator_guides,
    )

    scene.initial_guides = initial_guides
    scene.cluster_guides = cluster_guides

guide_roots = initial_guides[:, 0].detach().clone()
scene.guide_roots = guide_roots

if save_results:
    torch.save(
        {
            "guides": initial_guides,
            "strands": grooming_chain(
                roots, initial_guides, head.scalp_get_normal(initial_guides[:, 0])
            ),
            "chain_state_dict": grooming_chain.state_dict(),
        },
        os.path.join(out_dir, "stage1_initialization.pt"),
    )

# %%
# -------------------------------------------------------------------------------------------------
# Stage 2: Instancing Optimization
# -------------------------------------------------------------------------------------------------
print("Stage 2: Optimizing instancing weights...")
if not is_synthetic:
    losses, groom_opt = optimize_instancing(
        scene,
        lr=INSTANCING_LR,
        n_iterations=INSTANCING_N_ITERATIONS,
    )

    if save_results:
        torch.save(
            {
                "guides": cluster_guides,
                "strands": groom_opt,
                "chain_state_dict": grooming_chain.state_dict(),
            },
            os.path.join(out_dir, "stage2_instancing.pt"),
        )

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(losses)
    ax.set_title("Instancing Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    if save_results:
        plt.savefig(
            os.path.join(out_dir, "stage2_instancing_loss.png"), bbox_inches="tight"
        )

# %%
# -------------------------------------------------------------------------------------------------
# Stage 3: Guides Optimization
# -------------------------------------------------------------------------------------------------
all_losses = []
all_l2_losses = []
all_op_values = defaultdict(list)

print("Stage 3: Optimizing guides...")
guide_strands = Strands(
    StrandParameterization.CURVATURE_LP,
    initial_guides,
    head.scalp_uv_mapping(guide_roots),
    intra_weight=0,
    inter_weight=SMOOTHING_LAMBDA,
)
scene.guide_strands = guide_strands

losses, l2_losses, groom_opt, guides_opt = optimize_guides(
    scene,
    curv_lr=GUIDES_CURV_LR,
    length_lr=GUIDES_LENGTH_LR,
    root_dir_lr=GUIDES_ROOT_DIR_LR,
    n_iterations=(
        GUIDES_N_ITERATIONS if not USE_KNOWN_GUIDES and not NO_GUIDE_OPT else 0
    ),
    sigma_d=SIGMA_D,
)

all_losses.extend(losses)
all_l2_losses.extend(l2_losses)

if save_results:
    torch.save(
        {
            "guides": guides_opt,
            "strands": groom_opt,
            "chain_state_dict": grooming_chain.state_dict(),
        },
        os.path.join(out_dir, f"stage3_guides.pt"),
    )

# %%
# -------------------------------------------------------------------------------------------------
# Stage 4: Operator Parameter Optimization
# -------------------------------------------------------------------------------------------------
print("Stage 4: Optimizing operator parameters...")
op_values, losses, l2_losses, groom_opt, guides_opt = optimize_operator_parameters(
    scene,
    lr=OPERATOR_LR,
    n_iterations=OPERATOR_N_ITERATIONS if not USE_KNOWN_PARAMS else 0,
)

all_losses.extend(losses)
all_l2_losses.extend(l2_losses)
for key, value in op_values.items():
    all_op_values[key].extend(value)

if save_results:
    torch.save(
        {
            "guides": guides_opt,
            "strands": groom_opt,
            "chain_state_dict": grooming_chain.state_dict(),
        },
        os.path.join(out_dir, f"stage4_operator.pt"),
    )

# %%
# -------------------------------------------------------------------------------------------------
# Stage 5: Operator Random Optimization
# -------------------------------------------------------------------------------------------------
print("Stage 5: Optimizing operator random numbers...")
has_rand_params = len(grooming_chain.get_optimizable_parameters({"random": True})) > 0
if has_rand_params:
    op_values, losses, l2_losses, groom_opt, guides_opt = optimize_operator_random(
        scene,
        lr=OPERATOR_RAND_LR,
        n_iterations=OPERATOR_RAND_N_ITERATIONS,
        noise_factor_begin=OPERATOR_RAND_NOISE_FACTOR_BEGIN,
        noise_factor_end=OPERATOR_RAND_NOISE_FACTOR_END,
    )

    all_losses.extend(losses)
    all_l2_losses.extend(l2_losses)
    for key, value in op_values.items():
        all_op_values[key].extend(value)

    if save_results:
        torch.save(
            {
                "guides": guides_opt,
                "strands": groom_opt,
                "chain_state_dict": grooming_chain.state_dict(),
            },
            os.path.join(out_dir, f"stage5_rand.pt"),
        )

# %%
# -------------------------------------------------------------------------------------------------
# Final Guide Reassignment
# -------------------------------------------------------------------------------------------------
if not is_synthetic:
    print("Reassigning operator guides as needed...")
    reassign_operator_guides(
        scene,
        k_closest_op_guides=REASSIGN_GUIDES_K,
        head_dist_threshold=REASSIGN_GUIDES_THRESHOLD,
    )

# -------------------------------------------------------------------------------------------------
# Final Groom Rendering
# -------------------------------------------------------------------------------------------------

with torch.no_grad():
    guides_opt = guide_strands.compute_strands(guide_roots)
    groom_opt = grooming_chain(roots, guides_opt, head.scalp_get_normal(guide_roots))

    if save_results:
        torch.save(
            {
                "guides": guides_opt,
                "strands": groom_opt,
                "chain_state_dict": grooming_chain.state_dict(),
            },
            os.path.join(out_dir, "final.pt"),
        )

    render(
        {
            "strands": target_groom,
            "title": "Reference",
        },
        {
            "strands": groom_opt,
            "title": "Ours (Groom)",
        },
        {
            "strands": guides_opt,
            "title": "Ours (Guides)",
        },
        save_name="final",
    )

    groom_opt = grooming_chain(
        roots, guides_opt, head.scalp_get_normal(guide_roots), mean=True
    )
    render(
        {
            "strands": groom_opt,
            "title": "Mean Groom",
        },
        {
            "strands": guides_opt,
            "title": "Mean Guides",
        },
        save_name="final_mean",
    )


# %%
# -------------------------------------------------------------------------------------------------
# Plotting and saving results
# -------------------------------------------------------------------------------------------------
def find_target_value(target_name):
    if not is_synthetic:
        return None

    op_set = {}
    for operator in grooming_chain_target.operators:
        op_name = operator._get_name()

        if op_name not in op_set:
            op_set[op_name] = 1
        else:
            op_set[op_name] += 1
            op_name = f"{op_name}_{op_set[op_name]}"

        for name, value in operator.named_parameters():
            if f"{op_name}/{name}" == target_name:
                return value.item() if value.numel() == 1 else value

    return None


all_op_errors = {}

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].plot(all_losses)
ax[0].set_title("Loss")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Error")
ax[0].set_yscale("log")

ax[1].plot(all_l2_losses)
ax[1].set_title("L2 Loss")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Error")
ax[1].set_yscale("log")

n_colors = len(all_op_values)
cm = plt.get_cmap("tab20")
ax[2].set_prop_cycle(color=[cm(1.0 * i / 12) for i in range(n_colors)])
ax[3].set_prop_cycle(color=[cm(1.0 * i / 12) for i in range(n_colors)])

for i, (name, value) in enumerate(all_op_values.items()):
    if torch.is_tensor(value[-1]):
        continue
    print(name, f"{value[-1]:.4g}")
    ax[2].plot(value, label=name)
ax[2].set_title("Operator Parameters")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Value")

if is_synthetic:
    for i, (name, value) in enumerate(all_op_values.items()):
        if torch.is_tensor(value[-1]):
            continue

        value = torch.tensor(value)
        true_value = find_target_value(name)
        if true_value is None:
            err = torch.abs(value - 0)
        elif name == "Bend/angle":
            err = torch.minimum(
                torch.abs(value - true_value),
                torch.abs(value + true_value),
            )
        else:
            err = torch.abs(value - true_value)
        ax[3].plot(err, label=name)
        all_op_errors[name] = err

    for i, (name, value) in enumerate(all_op_values.items()):
        if not torch.is_tensor(value[-1]):
            continue

        value = torch.stack(value)
        true_value = find_target_value(name)
        if true_value is None:
            err = torch.abs(value - 0).mean(dim=(1, 2, 3)).squeeze().detach().cpu()
        else:
            if name == "Curl/rand2":
                err = (
                    torch.abs(
                        value % (1 / 3.9) * 3.9 - true_value[None] % (1 / 3.9) * 3.9
                    )
                    .mean(dim=(1, 2, 3))
                    .squeeze()
                    .detach()
                    .cpu()
                )
            else:
                err = (
                    torch.abs(value - true_value[None])
                    .mean(dim=(1, 2, 3))
                    .squeeze()
                    .detach()
                    .cpu()
                )
        ax[3].plot(err, label=name)
        all_op_errors[name] = err

    ax[3].legend(loc="upper right", bbox_to_anchor=(1.55, 1.1))
    ax[3].set_title("Operator Parameter Errors")
    ax[3].set_xlabel("Iteration")
    ax[3].set_ylabel("MAE")
plt.tight_layout()
if save_results:
    plt.savefig(os.path.join(out_dir, "training.png"), bbox_inches="tight")

if save_results:
    torch.save(
        {
            "op_values": all_op_values,
            "op_errors": all_op_errors,
            "losses": all_losses,
            "l2_losses": all_l2_losses,
        },
        os.path.join(out_dir, "training.pt"),
    )

    np.savez(
        os.path.join(out_dir, "final.npz"),
        groom=head.scale_from_head(
            grooming_chain(roots, guides_opt, head.scalp_get_normal(guide_roots))
        )
        .detach()
        .cpu()
        .numpy(),
    )
