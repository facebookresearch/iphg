# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from scene import Scene


def add_to_op_values_dict(scene: Scene, op_values: Dict[str, List[float]]):
    grooming_chain = scene.grooming_chain
    is_synthetic = scene.is_synthetic
    initial_guides = scene.initial_guides
    target_guides = scene.target_guides

    track_rand_params = (
        is_synthetic and initial_guides.shape[0] == target_guides.shape[0]
    )

    op_set = {}
    for operator in grooming_chain.operators:
        op_name = operator._get_name()

        if op_name not in op_set:
            op_set[op_name] = 1
        else:
            op_set[op_name] += 1
            op_name = f"{op_name}_{op_set[op_name]}"

        for name, value in operator.named_parameters():
            if value.requires_grad and (value.numel() == 1 or track_rand_params):
                val = value.item() if value.numel() == 1 else value.clone()
                op_values[f"{op_name}/{name}"].append(val)
