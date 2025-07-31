import os
import torch

from grooming_graph.operators import (
    OperatorChain,
    Scale,
    Curl,
    Bend,
    Clump,
    Frizz,
    RandMode,
)
from grooming_graph.operators.guides import Instance
from head import HeadModel
from dataset_readers import load_synthetic_groom_guides

default_view = 14


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
                max_guides_per_root=3,
                optimizable=False,
            ),
            Clump(0.0),
            Bend(
                0.0,
                0.5,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
            Frizz(
                0.5,
                0.0,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
            Scale(
                1.0,
                0.0,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
        ],
        generator=gen,
    )
    return grooming_chain


def initialize_target_grooming_chain(
    n_strands, n_guides, head: HeadModel, seed=0, rand_mode=RandMode.OPTIMIZE
):
    gen_target = torch.Generator(device="cuda")
    gen_target.manual_seed(seed)
    grooming_chain = OperatorChain(
        [
            Instance(
                0.05,
                n_strands,
                roots_to_uv=head.scalp_uv_mapping,
                max_guides_per_root=3,
                optimizable=False,
            ),
            Clump(0.5),
            Bend(
                0.6,
                0.6,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
            Frizz(
                1.0,
                0.05,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
            Scale(
                1.0,
                0.2,
                n_strands=n_strands,
                rand_mode=rand_mode,
            ),
        ],
        generator=gen_target,
    )
    return grooming_chain


def load_groom_guides(head: HeadModel):
    dirname = os.path.abspath(os.path.dirname(__file__))

    n_guides = 500
    target_guides, target_groom = load_synthetic_groom_guides(head, dirname)

    rand_guide_idx = torch.randperm(len(target_guides))
    target_guides = target_guides[rand_guide_idx[: int(len(target_guides) / 2)]]

    return n_guides, target_guides, target_groom
