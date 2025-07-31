# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from collections import defaultdict
from tqdm import tqdm
from scene import Scene
from optimization.loss import rand_loss
from optimization.utils import add_to_op_values_dict


def lerp(a, b, t):
    return a + (b - a) * t


def optimize_operator_random(
    scene: Scene,
    lr: float = 5e-2,
    n_iterations: int = 3000,
    noise_factor_begin: float = 5.0,
    noise_factor_end: float = 0.0,
):
    grooming_chain = scene.grooming_chain
    head = scene.head
    target_groom = scene.target_groom
    roots = scene.roots
    opt_strand_idx = scene.opt_strand_idx
    guide_roots = scene.guide_roots
    render = scene.render_fn
    guide_strands = scene.guide_strands

    random_parameters = grooming_chain.get_optimizable_parameters({"random": True})

    opt = torch.optim.Adam(params=random_parameters, lr=lr, eps=1e-15)

    op_values = defaultdict(list)
    losses = []
    l2_losses = []

    progress_bar = tqdm(range(n_iterations))

    for it in range(1, n_iterations + 1):
        add_to_op_values_dict(scene, op_values)

        opt.zero_grad()

        guides_opt = guide_strands.compute_strands(guide_roots)
        groom_opt = grooming_chain(
            roots, guides_opt, head.scalp_get_normal(guide_roots)
        )

        loss = rand_loss(groom_opt[opt_strand_idx], target_groom)
        loss.backward()

        w = (it - 1) / (n_iterations - 1)
        for op in grooming_chain.operators:
            for param in op.parameters():
                if param.is_random():
                    param.grad += (
                        torch.randn_like(param)
                        * param.grad.abs().mean()
                        * lerp(noise_factor_begin, noise_factor_end, w)
                    )

        opt.step()

        with torch.no_grad():
            for operator in grooming_chain.operators:
                operator.clamp_parameters()

        l2_loss = (groom_opt[opt_strand_idx] - target_groom).square().mean()

        losses.append(loss.item())
        l2_losses.append(l2_loss.item())

        if it % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{l2_loss.item():.4g}"})
            progress_bar.update(10)

    progress_bar.close()

    with torch.no_grad():
        guides_opt = guide_strands.compute_strands(guide_roots)
        groom_opt = grooming_chain(
            roots,
            guides_opt,
            head.scalp_get_normal(guide_roots),
        )

        render(
            {
                "strands": groom_opt,
                "title": f"Rand Optimized Groom",
            },
            {
                "strands": guides_opt,
                "title": f"Rand Optimized Guides",
            },
            save_name="stage5_rand",
        )

    return op_values, losses, l2_losses, groom_opt, guides_opt
