import os
import torch

from grooming_graph.operators import OperatorChain, Frizz, Curl, Bend, Clump, RandMode
from grooming_graph.operators.guides import Instance
from head import HeadModel
from dataset_readers import load_synthetic_groom_guides


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
                max_guides_per_root=1,
                optimizable=False,
            ),
            Curl(
                1.0,
                0.05,
                0.5,
                rand_frequency=0.0,
                curl_start=0.0,
                n_strands=n_strands,
                n_guides=n_guides,
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
                max_guides_per_root=1,
                optimizable=False,
            ),
            Curl(
                1.0,
                0.06,
                0.8,
                rand_frequency=0.5,
                curl_start=0.3,
                n_strands=n_strands,
                n_guides=n_guides,
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

    return n_guides, target_guides, target_groom
