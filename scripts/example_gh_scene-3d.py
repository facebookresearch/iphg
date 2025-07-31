# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from grooming_graph.operators import (
    OperatorChain,
    Frizz,
    Scale,
    Bend,
    Clump,
    RandMode,
)
from grooming_graph.operators.guides import Instance
from head import HeadModel
from dataset_readers import load_gaussian_haircut_groom


def initialize_grooming_chain(
    n_strands, n_guides, head: HeadModel, seed=0, rand_mode=RandMode.OPTIMIZE
):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    grooming_chain = OperatorChain(
        [
            Instance(
                0.05,
                n_strands,
                roots_to_uv=head.scalp_uv_mapping,
                max_guides_per_root=4,
            ),
            Frizz(
                0.01,
                0.01,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
        ],
        generator=gen,
    )
    return grooming_chain


def load_groom_guides(head: HeadModel):
    dirname = os.path.abspath(os.path.dirname(__file__))

    n_guides = 500

    target_groom = load_gaussian_haircut_groom(head, dirname)

    return n_guides, None, target_groom
