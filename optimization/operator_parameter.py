# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from collections import defaultdict
from tqdm import tqdm
from enum import IntEnum
from scene import Scene
from optimization.loss import shape_loss, clump_loss, scale_loss, RandomStrandSelector
from optimization.utils import add_to_op_values_dict


class OperatorIterType(IntEnum):
    SHAPE = 0
    CLUMP = 1
    SCALE = 2


def optimize_operator_parameters(
    scene: Scene,
    lr: float = 1e-2,
    n_iterations: int = 3000,
    op_interval: int = 200,
):
    grooming_chain = scene.grooming_chain
    head = scene.head
    is_synthetic = scene.is_synthetic
    target_groom = scene.target_groom
    roots = scene.roots
    opt_strand_idx = scene.opt_strand_idx
    guide_roots = scene.guide_roots
    render = scene.render_fn
    guide_strands = scene.guide_strands

    has_clump = len(grooming_chain.get_optimizable_parameters({"Clump": True})) > 0
    has_scale = len(grooming_chain.get_optimizable_parameters({"Scale": True})) > 0
    has_shape = (
        len(
            grooming_chain.get_optimizable_parameters(
                {
                    "Bend": True,
                    "Curl": True,
                    "Frizz": True,
                }
            )
        )
        > 0
    )

    non_clump_parameters = grooming_chain.get_optimizable_parameters({"Clump": False})
    non_scale_parameters = grooming_chain.get_optimizable_parameters({"Scale": False})
    non_shape_parameters = grooming_chain.get_optimizable_parameters(
        {
            "Bend": False,
            "Curl": False,
            "Frizz": False,
        }
    )

    opt = torch.optim.Adam(params=grooming_chain.get_optimizable_parameters(), lr=lr)

    def get_op_iter_type(it, interval=op_interval):
        iter_types = (
            [OperatorIterType.SHAPE] * has_shape
            + [OperatorIterType.CLUMP] * has_clump
            + [OperatorIterType.SCALE] * has_scale
        )
        it = it - 1
        iter_type_idx = (it // interval) % len(iter_types)
        return iter_types[iter_type_idx]

    op_values = defaultdict(list)
    losses = []
    l2_losses = []

    rss = RandomStrandSelector(head, roots[opt_strand_idx])

    n_iterations = n_iterations * (has_shape + has_clump + has_scale)
    progress_bar = tqdm(range(n_iterations))

    for it in range(1, n_iterations + 1):
        add_to_op_values_dict(scene, op_values)

        op_iter_type = get_op_iter_type(it)

        opt.zero_grad()

        guides_opt = guide_strands.compute_strands(guide_roots)
        groom_opt = grooming_chain(
            roots, guides_opt, head.scalp_get_normal(guide_roots)
        )

        if is_synthetic:
            n_total_samples = 10000 if (it - 1) / (n_iterations - 1) >= 0.75 else 1000
        else:
            n_total_samples = 6000 if (it - 1) / (n_iterations - 1) >= 0.75 else 2000

        match op_iter_type:
            case OperatorIterType.SHAPE:
                loss = shape_loss(groom_opt[opt_strand_idx], target_groom)
            case OperatorIterType.CLUMP:
                loss = clump_loss(
                    rss,
                    groom_opt[opt_strand_idx],
                    target_groom,
                    n_total_samples=n_total_samples,
                )
            case OperatorIterType.SCALE:
                loss = scale_loss(
                    rss,
                    groom_opt[opt_strand_idx],
                    target_groom,
                    n_total_samples=n_total_samples,
                )

        loss.backward()

        match op_iter_type:
            case OperatorIterType.SHAPE:
                for param in non_shape_parameters:
                    param.grad = None
            case OperatorIterType.CLUMP:
                for param in non_clump_parameters:
                    param.grad = None
            case OperatorIterType.SCALE:
                for param in non_scale_parameters:
                    param.grad = None

        opt.step()

        with torch.no_grad():
            for operator in grooming_chain.operators:
                operator.clamp_parameters()

        l2_loss = (groom_opt[opt_strand_idx] - target_groom).square().mean()

        losses.append(loss.item())
        l2_losses.append(l2_loss.item())

        if it % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4g}"})
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
                "title": f"Operator Optimized Groom",
            },
            {
                "strands": guides_opt,
                "title": f"Operator Optimized Guides",
            },
            save_name="stage4_operator",
        )

    return op_values, losses, l2_losses, groom_opt, guides_opt
