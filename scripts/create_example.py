# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import shutil
from argparse import ArgumentParser

from convert_monohair import convert_head, convert_hair

cur_dir = os.path.dirname(os.path.abspath(__file__))

parser = ArgumentParser()
parser.add_argument("--type", type=str, required=True, choices=["gh", "mh"])
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--example", type=str, required=True)
args, _ = parser.parse_known_args(sys.argv[1:])

if not os.path.exists(args.in_dir):
    raise FileNotFoundError(f"Input directory {args.in_dir} does not exist.")

if args.type == "gh":
    gh_out_dir = os.path.abspath(
        os.path.join(cur_dir, "..", "data", f"{args.example}_gh")
    )
    if os.path.exists(gh_out_dir):
        print(
            f"Output directory {gh_out_dir} already exists. Please remove it or choose a different example name."
        )
        sys.exit(1)

    os.makedirs(gh_out_dir)
    print(f"Creating in directory: {gh_out_dir}")

    shutil.copy(
        os.path.join(
            args.in_dir,
            "3d_gaussian_splatting",
            "stage1",
            "cameras",
            "30000_matrices.pkl",
        ),
        os.path.join(gh_out_dir, "cameras.pkl"),
    )
    shutil.copy(
        os.path.join(
            args.in_dir, "flame_fitting", "stage1", "stage_3", "mesh_final.obj"
        ),
        os.path.join(gh_out_dir, "head_flame.obj"),
    )
    shutil.copy(
        os.path.join(
            args.in_dir, "curves_reconstruction", "stage3", "blender", "hair.npy"
        ),
        os.path.join(gh_out_dir, "strands.npy"),
    )

    shutil.copy(
        os.path.join(cur_dir, "example_gh_scene-3d.py"),
        os.path.join(gh_out_dir, "scene-3d.py"),
    )
elif args.type == "mh":
    mh_out_dir = os.path.abspath(
        os.path.join(cur_dir, "..", "data", f"{args.example}_mh")
    )
    if os.path.exists(mh_out_dir):
        print(
            f"Output directory {mh_out_dir} already exists. Please remove it or choose a different example name."
        )
        sys.exit(1)

    os.makedirs(mh_out_dir)
    print(f"Creating in directory: {mh_out_dir}")

    convert_head(args.in_dir, mh_out_dir)
    convert_hair(args.in_dir, mh_out_dir)

    shutil.copy(
        os.path.join(
            args.in_dir,
            "ours",
            "cam_params.json",
        ),
        os.path.join(mh_out_dir, "cam_params.json"),
    )
    shutil.copy(
        os.path.join(args.in_dir, "model_tsfm.dat"),
        os.path.join(mh_out_dir, "model_tsfm.dat"),
    )

    shutil.copy(
        os.path.join(cur_dir, "example_mh_scene-3d.py"),
        os.path.join(mh_out_dir, "scene-3d.py"),
    )
