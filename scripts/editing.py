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
from copy import deepcopy

module_path = os.path.abspath(os.path.join(__file__, "..", ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from head import HeadModel
from grooming_graph.utils.hair import resample_strands
from grooming_graph.operators.guides import BoundedParameter
from grooming_graph.operators import (
    Scale,
    Bend,
    Frizz,
    RandMode,
)


torch.manual_seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)

parser = ArgumentParser()
parser.add_argument(
    "--blender_path",
    type=str,
    default="~/.local/share/flatpak/app/org.blender.Blender/x86_64/stable/active/files/blender/blender",
)
args, _ = parser.parse_known_args(sys.argv[1:])

dataset = "synthetic-grooms-coily"
view = 28
results_dir = os.path.join(module_path, "output", dataset)

if not os.path.exists(results_dir):
    raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

exp_num = sorted([int(d) for d in os.listdir(results_dir)])[-1]

results_dir = os.path.join(results_dir, str(exp_num))

if not os.path.exists(results_dir):
    raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

dataset_path = os.path.join(module_path, "data", dataset)

scene_file_path = os.path.join(dataset_path, "scene-3d.py")
scene_dict = {"__file__": scene_file_path}
with open(scene_file_path) as f:
    exec(f.read(), scene_dict)

head = HeadModel(dataset_path, scene_dict.get("transform", None))

n_guides, target_guides, target_groom = scene_dict["load_groom_guides"](head)

roots = target_groom[:, 0].clone()
roots = head.resample_roots(roots, int(len(roots) * 1.25))

n_strands = roots.shape[0]

grooming_chain = scene_dict["initialize_grooming_chain"](
    n_strands,
    n_guides,
    head=head,
    seed=0,
)
grooming_chain.to(target_groom.device)

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

print("Operator order:")
for i, op in enumerate(grooming_chain.operators):
    print(f"{i}: {op._get_name()}")
print()

print("Operator parameters:")
for op in grooming_chain.operators:
    op_name = op._get_name()
    for name, param in op.named_parameters():
        if param.numel() == 1:
            print(f"{op_name}/{name}: {param.item()}")
print()

our_groom = grooming_chain(roots, our_guides, head.scalp_get_normal(our_guides[:, 0]))

# Guide editing
guides_path = os.path.join(dataset_path, "guides.npy")
guides_long = torch.from_numpy(np.load(guides_path)).cuda()
guides_long = head.scale_to_head(guides_long)
guides_long = head.project_strands_onto_scalp(guides_long)
edited_guides = resample_strands(guides_long, our_guides.shape[1])

bend = Bend(
    0.35,
    0.0,
    n_strands=edited_guides.shape[0],
    rand_mode=RandMode.FIX,
    optimizable=False,
).cuda()
edited_guides = bend(edited_guides, head.scalp_get_normal(edited_guides[:, 0]))

# Since the number of guides changed, need to reinitialize the grooming chain
# Also remove any operator parameters that depend on the number of guides
grooming_chain = scene_dict["initialize_grooming_chain"](
    roots.shape[0], edited_guides.shape[0], head=head
)
grooming_chain.to(target_groom.device)
chain_dict = deepcopy(our_data["chain_state_dict"])
for op_dict in chain_dict["operators"]:
    for k, v in list(op_dict.items()):
        if (isinstance(v, BoundedParameter) or isinstance(v, torch.Tensor)) and (
            len(v.shape) > 0 and v.shape[0] == n_guides
        ):
            op_dict.pop(k)

grooming_chain.load_state_dict(chain_dict, strict=False)

# Add bend operator to the chain
grooming_chain.operators.append(
    Bend(
        0.6,
        0.25,
        n_strands=roots.shape[0],
    ).to(target_groom.device)
)

edited_groom1 = grooming_chain(
    roots, edited_guides, head.scalp_get_normal(edited_guides[:, 0])
)

# Remove curls and bend, add some Frizz, SetLength
grooming_chain.operators[2].curl_factor[...] = 0.0
grooming_chain.operators[3].angle[...] = 0.5
grooming_chain.operators.append(
    Frizz(
        0.5,
        0.05,
        n_strands=roots.shape[0],
    ).to(target_groom.device)
)
grooming_chain.operators.append(
    Scale(
        1.0,
        0.1,
        n_strands=roots.shape[0],
    ).to(target_groom.device)
)

edited_groom2 = grooming_chain(
    roots, edited_guides, head.scalp_get_normal(edited_guides[:, 0])
)


np.savez(
    os.path.join(results_dir, "final.npz"),
    groom=head.scale_from_head(our_groom).cpu().numpy(),
    target_groom=head.scale_from_head(target_groom).cpu().numpy(),
    edited_groom1=head.scale_from_head(edited_groom1).cpu().numpy(),
    edited_groom2=head.scale_from_head(edited_groom2).cpu().numpy(),
)

blender_script_path = os.path.join(module_path, "scripts", "blender")
render_script = "render_synthetic.py"
command = (
    f"{args.blender_path} -b {os.path.join(blender_script_path, 'render.blend')} -P {os.path.join(blender_script_path, render_script)} -- --args "
    + f"{dataset_path} {dataset} {results_dir} {results_dir} {view} True"
)

run(command, shell=True)
