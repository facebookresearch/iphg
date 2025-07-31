# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# %%
import os
import sys
import torch
import numpy as np
from subprocess import run
from argparse import ArgumentParser
import pickle

module_path = os.path.abspath(os.path.join(__file__, "..", ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from head import HeadModel
from dataset_readers import load_K_Rt_from_P

torch.manual_seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)

parser = ArgumentParser()
parser.add_argument(
    "--blender_path",
    type=str,
    default="~/.local/share/flatpak/app/org.blender.Blender/x86_64/stable/active/files/blender/blender",
)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--exp_num", type=int, default=-1)
parser.add_argument("--view", type=int, default=21)
args, _ = parser.parse_known_args(sys.argv[1:])

dataset = args.dataset
results_dir = os.path.join(module_path, "output", dataset)

if not os.path.exists(results_dir):
    raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

exp_num = args.exp_num
if exp_num == -1:
    exp_num = sorted([int(d) for d in os.listdir(results_dir)])[-1]

results_dir = os.path.join(results_dir, str(exp_num))

if not os.path.exists(results_dir):
    raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

dataset_path = os.path.join(module_path, "data", dataset)
is_synthetic = "synthetic" in dataset
is_gh = "gh" in dataset

scene_file_path = os.path.join(dataset_path, "scene-3d.py")
scene_dict = {"__file__": scene_file_path}
with open(scene_file_path) as f:
    exec(f.read(), scene_dict)

head = HeadModel(dataset_path, scene_dict.get("transform", None))

n_guides, target_guides, target_groom = scene_dict["load_groom_guides"](head)

roots = target_groom[:, 0].clone()
if is_synthetic:
    roots = head.resample_roots(roots, int(len(roots) * 1.25))

n_strands = roots.shape[0]

grooming_chain = scene_dict["initialize_grooming_chain"](
    n_strands,
    n_guides,
    head=head,
    seed=0,
)
grooming_chain.to(target_groom.device)

if is_synthetic:
    grooming_chain_target = scene_dict["initialize_target_grooming_chain"](
        n_strands, target_guides.shape[0], head=head, seed=0
    )
    grooming_chain_target.to(target_groom.device)

    target_groom = grooming_chain_target(
        roots, target_guides, head.scalp_get_normal(target_guides[:, 0])
    )

our_data = torch.load(os.path.join(results_dir, "final.pt"))
our_guides = our_data["guides"]

grooming_chain.load_state_dict(our_data["chain_state_dict"], strict=False)
our_groom = grooming_chain(roots, our_guides, head.scalp_get_normal(our_guides[:, 0]))

if is_gh:
    file_path = os.path.join(dataset_path, "cameras.pkl")
    camera_data = pickle.load(open(file_path, "rb"))

    cam_data = list(camera_data.values())[args.view]
    K, pose = load_K_Rt_from_P(cam_data.transpose(0, 1)[:3, :4].numpy())
    np.savez(
        os.path.join(results_dir, "camera.npz"),
        K=K,
        pose=pose,
    )

if is_synthetic:
    np.savez(
        os.path.join(results_dir, "final.npz"),
        groom=head.scale_from_head(our_groom).cpu().numpy(),
        guides=head.scale_from_head(our_guides).cpu().numpy(),
        target_guides=head.scale_from_head(target_guides).cpu().numpy(),
        target_groom=head.scale_from_head(target_groom).cpu().numpy(),
    )
else:
    np.savez(
        os.path.join(results_dir, "final.npz"),
        groom=head.scale_from_head(our_groom).cpu().numpy(),
        guides=head.scale_from_head(our_guides).cpu().numpy(),
        target_groom=head.scale_from_head(target_groom).cpu().numpy(),
    )

blender_script_path = os.path.join(module_path, "scripts", "blender")
render_script = "render_synthetic.py" if is_synthetic else "render_real.py"
command = (
    f"{args.blender_path} -b {os.path.join(blender_script_path, 'render.blend')} -P {os.path.join(blender_script_path, render_script)} -- --args "
    + f"{dataset_path} {dataset} {results_dir} {results_dir} {args.view}"
)

run(command, shell=True)
