# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from grooming_graph.operators import (
    OperatorChain,
    Scale,
    Curl,
    Clump,
    Frizz,
    RandMode,
)
from grooming_graph.operators.guides import Instance
from head import HeadModel
from dataset_readers import load_monohair_groom, load_monohair_transform

dirname = os.path.abspath(os.path.dirname(__file__))

transform, translation = load_monohair_transform(dirname)


def initialize_grooming_chain(
    n_strands, n_guides, head: HeadModel, seed=0, rand_mode=RandMode.OPTIMIZE
):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    grooming_chain = OperatorChain(
        [
            Instance(
                0.01,
                n_strands,
                operator_guide_ratio=1 / 4,
                roots_to_uv=head.scalp_uv_mapping,
                max_guides_per_root=4,
            ),
            Curl(
                1.0,
                0.05,
                0.5,
                n_strands=n_strands,
                n_guides=n_guides,
                rand_mode=rand_mode,
            ),
        ],
        generator=gen,
    )
    return grooming_chain


def load_groom_guides(head: HeadModel):
    n_guides = 500
    n_strand_segments = 99

    target_groom = load_monohair_groom(
        head, dirname, translation, n_strand_segments=n_strand_segments
    )

    return n_guides, None, target_groom
