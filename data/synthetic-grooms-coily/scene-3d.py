import os
import torch
import numpy as np
from sklearn.neighbors import KDTree

from grooming_graph.operators import (
    OperatorChain,
    Bend,
    Curl,
    Clump,
    RandMode,
)
from grooming_graph.operators.guides import Instance
from grooming_graph.utils.hair import resample_strands
from head import HeadModel

default_view = 14
smooth_steps = 2000


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
                optimizable=False,
            ),
            Clump(0.0, curl_clump=True),
            Curl(
                1.0,
                0.03,
                3.0,
                curl_start=0.2,
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
                max_guides_per_root=4,
                optimizable=False,
            ),
            Clump(1.5, curl_clump=True),
            Curl(
                1.0,
                0.02,
                15,
                curl_start=0.2,
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

    n_guides = 3000
    n_strand_segments = 100

    guides_path = os.path.join(dirname, "guides.npy")
    target_guides = torch.from_numpy(np.load(guides_path)).cuda()
    target_guides = head.scale_to_head(target_guides)
    target_guides = head.project_strands_onto_scalp(target_guides)

    # Make guides longer towards middle-back of head
    target_guide_uvs = head.scalp_uv_mapping(target_guides[:, 0])
    target_guide_uv_dist = (
        target_guide_uvs - torch.tensor([0.6, 0.4], device=target_guide_uvs.device)
    ).norm(dim=-1)
    target_guide_end_idx = (14 * (1 - target_guide_uv_dist / 0.5**0.5)).int()
    target_guides = resample_strands(
        target_guides, n_strand_segments + 1, ending_idx=target_guide_end_idx
    )

    bend = Bend(
        0.6,
        0.0,
        n_strands=target_guides.shape[0],
        rand_mode=RandMode.FIX,
        optimizable=False,
    ).cuda()
    target_guides = bend(target_guides, head.scalp_get_normal(target_guides[:, 0]))

    # Subsample target guides
    rand = torch.randperm(len(target_guides), device=target_guides.device)[:n_guides]
    target_guides = target_guides[rand]

    groom_path = os.path.join(dirname, "groom.npy")
    target_groom = torch.from_numpy(np.load(groom_path)).cuda()
    target_groom = head.scale_to_head(target_groom)
    target_groom = head.project_strands_onto_scalp(target_groom)
    target_groom = resample_strands(target_groom, n_strand_segments + 1)

    # Remove target groom strands that are too far from guides
    tree = KDTree(target_guides[:, 0].cpu().numpy())
    dist = tree.query(target_groom[:, 0].cpu().numpy(), k=1, return_distance=True)[
        0
    ].squeeze()
    target_groom = target_groom[dist < 0.03]

    return n_guides, target_guides, target_groom
