# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import struct
import torch
import numpy as np
import re

module_path = os.path.abspath(os.path.join(__file__, "..", ".."))

if module_path not in sys.path:
    sys.path.append(module_path)

from plyfile import PlyData, PlyElement
from grooming_graph.utils.hair import resample_strands


def convert_head(in_path, out_path):
    head_file = os.path.join(in_path, "ours", "Voxel_hair", "flame_bust.obj")
    head_out = os.path.join(out_path, "head_flame.obj")

    with open(head_file, "r") as f:
        lines = f.readlines()
        lines = [
            line for line in lines if line.startswith("v ") or line.startswith("f ")
        ]

        pattern = re.compile(r"f (\d+)(/\d+)+ (\d+)(/\d+)+ (\d+)(/\d+)+")
        lines = [pattern.sub(r"f \1 \3 \5", line) for line in lines]

    with open(head_out, "w") as f:
        f.writelines(lines)


def convert_hair(in_path, out_path, short_strand_threshold=10):
    hair_file = os.path.join(
        in_path, "output", "10-16", "full", "connected_strands.hair"
    )
    hair_out = os.path.join(out_path, "strands.ply")

    n_strand_segments = 99

    with open(hair_file, "rb") as f:
        #
        num_strand = f.read(4)
        (num_strand,) = struct.unpack("I", num_strand)
        point_count = f.read(4)
        (point_count,) = struct.unpack("I", point_count)
        #
        print("num_strand:", num_strand)
        print("point_count:", point_count)
        #
        segments = f.read(2 * num_strand)
        segments = struct.unpack("H" * num_strand, segments)
        segments = list(segments)

        #
        num_points = sum(segments)
        assert num_points == point_count
        #
        points = f.read(4 * num_points * 3)
        points = struct.unpack("f" * num_points * 3, points)
        points = list(points)
        points = torch.tensor(points).reshape(-1, 3)

    strands = []
    point_index_start = 0
    for strand_vertex_count in segments:
        #
        point_index_end = point_index_start + strand_vertex_count
        #
        if strand_vertex_count >= short_strand_threshold:
            strand = points[point_index_start:point_index_end]
            strand = resample_strands(strand.view(1, -1, 3), n_strand_segments + 1)
            strands.append(strand)
        #
        point_index_start += strand_vertex_count

    strands = torch.cat(strands, dim=0)

    print(f"Converted {len(strands)} strands with {n_strand_segments} segments each.")

    strands = strands.reshape(-1, 3).numpy()
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    elements = np.empty(strands.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, strands))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(hair_out)
