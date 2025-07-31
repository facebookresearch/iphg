# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import warp as wp
from typing import Callable, Optional
from .operator import Operator, BoundedParameter, RandomParameter

from grooming_graph.utils.knn import knn


class Instance(Operator):
    def __init__(
        self,
        roi_sigma: float,
        n_strands: int,
        operator_guide_jitter: Optional[float] = 0.5,
        operator_guide_ratio=1.0,
        roots_to_uv: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        max_guides_per_root: int = 5,
        optimizable: bool = True,
    ):
        super().__init__()
        self.roi_sigma = roi_sigma
        self.roots_to_uv = roots_to_uv
        self.max_guides_per_root = max_guides_per_root
        self.operator_guide_jitter = BoundedParameter(
            operator_guide_jitter, min_value=0.0, requires_grad=False
        )
        self.operator_guide_ratio = operator_guide_ratio
        self.rand = RandomParameter(
            (n_strands, 2),
            distribution="normal",
            requires_grad=False,
        )
        self.uv_rand_offset = None
        self.roi_weights = BoundedParameter(
            torch.ones(n_strands, max_guides_per_root),
            min_value=0,
            requires_grad=optimizable,
        )
        self.operator_guide_idxs = None
        self.operator_guide_assignment = None
        self.guide_assignment = None

        self.hg = wp.HashGrid(
            128,
            128,
            1,
            device="cuda",
        )
        self.operator_hg = wp.HashGrid(
            64,
            64,
            1,
            device="cuda",
        )
        self.search_radius = 0.7071  # sqrt(0.5), half the diagonal of the texture map
        self.operator_search_radius = 0.7071
        self.search_radius_initialized = False

    def state_dict(self):
        state_dict = super().state_dict()
        return {
            **state_dict,
            "operator_guide_idxs": self.operator_guide_idxs,
            "operator_guide_assignment": self.operator_guide_assignment,
            "guide_assignment": self.guide_assignment,
            "uv_rand_offset": self.uv_rand_offset,
            "search_radius": self.search_radius,
            "operator_search_radius": self.operator_search_radius,
            "search_radius_initialized": self.search_radius_initialized,
        }

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()
        if "operator_guide_idxs" in state_dict:
            self.operator_guide_idxs = state_dict.pop("operator_guide_idxs")
        if "operator_guide_assignment" in state_dict:
            self.operator_guide_assignment = state_dict.pop("operator_guide_assignment")
        if "guide_assignment" in state_dict:
            self.guide_assignment = state_dict.pop("guide_assignment")
        if "uv_rand_offset" in state_dict:
            self.uv_rand_offset = state_dict.pop("uv_rand_offset")
        if "search_radius" in state_dict:
            self.search_radius = state_dict.pop("search_radius")
        if "operator_search_radius" in state_dict:
            self.operator_search_radius = state_dict.pop("operator_search_radius")
        if "search_radius_initialized" in state_dict:
            self.search_radius_initialized = state_dict.pop("search_radius_initialized")
        return super().load_state_dict(state_dict, strict)

    @torch.no_grad()
    def update(self, roots: torch.Tensor, guides: torch.Tensor):
        guide_roots = guides[:, 0]
        guide_roots_uv = self.roots_to_uv(guide_roots)
        guide_roots_points = torch.cat(
            (guide_roots_uv, torch.zeros_like(guide_roots_uv[:, 0:1])),
            dim=1,
        )
        points = wp.from_torch(guide_roots_points, dtype=wp.vec3)
        self.hg.build(points, self.search_radius)

        if self.operator_guide_idxs is None:
            if self.operator_guide_ratio == 1.0:
                # WTODO: Might be a bug on why this branch is needed
                self.operator_guide_idxs = torch.arange(
                    len(guides), device=guides.device
                )
            else:
                n_operator_guides = int(len(guides) * self.operator_guide_ratio)
                self.operator_guide_idxs = torch.randperm(
                    len(guides), generator=self.generator, device=guides.device
                )[:n_operator_guides]

        points = wp.from_torch(
            guide_roots_points[self.operator_guide_idxs], dtype=wp.vec3
        )
        self.operator_hg.build(points, self.operator_search_radius)

        roots_uv = self.roots_to_uv(roots)
        if not self.search_radius_initialized:
            _, dists = knn(
                self.hg.id,
                roots_uv,
                guide_roots_uv,
                self.max_guides_per_root,
                self.search_radius,
                False,
            )

            def roi_func(dists2) -> torch.Tensor:
                return torch.exp(-dists2 / (self.roi_sigma**2)) + 1e-5

            roi_weights = roi_func(dists**2)
            roi_weights[dists > 1] = (
                0  # Set weights to 0 if there are no guides within the search radius
            )
            self.roi_weights[...] = roi_weights

            dists = dists[
                dists < 1
            ]  # Remove distances if there are no guides within the search radius
            self.search_radius = dists.max().item() * 0.5 * 2

            _, dists = knn(
                self.operator_hg.id,
                roots_uv,
                guide_roots_uv[self.operator_guide_idxs],
                1,
                self.operator_search_radius,
                False,
            )
            dists = dists[
                dists < 1
            ]  # Remove distances if there are no guides within the search radius
            self.operator_search_radius = dists.max().item() * 0.5 * 2

            self.search_radius_initialized = True

        if (
            self.uv_rand_offset is None
            or self.uv_rand_offset.shape[0] != roots_uv.shape[0]
        ):
            _, dists = knn(
                self.operator_hg.id,
                guide_roots_uv,
                guide_roots_uv,
                1,
                self.operator_search_radius,
                True,
            )
            avg_guide_root_dist = dists[dists < 1].mean()
            avg_root_dist = avg_guide_root_dist / (len(roots_uv) / len(guide_roots_uv))
            self.uv_rand_offset = self.rand * avg_root_dist * self.operator_guide_jitter

    def compute_blended_guides(
        self, roots: torch.Tensor, guides: torch.Tensor
    ) -> torch.Tensor:
        guide_roots = guides[:, 0]  # (M, 3), self.roots (N, 3)
        guide_roots_uv = self.roots_to_uv(guide_roots)
        roots_uv = self.roots_to_uv(roots)

        guide_idx, _ = knn(
            self.hg.id,
            roots_uv,
            guide_roots_uv,
            self.max_guides_per_root,
            self.search_radius,
            False,
        )
        roi_weights_sum = self.roi_weights.sum(dim=-1, keepdim=True) + 1e-5
        roi_weights = self.roi_weights / roi_weights_sum

        nearest_guides = guides[guide_idx.flatten()].reshape(
            guide_idx.shape[0], guide_idx.shape[1], guides.shape[1], guides.shape[2]
        )
        blended_guides = (
            roi_weights.view(-1, self.max_guides_per_root, 1, 1) * nearest_guides
        ).sum(dim=1)

        if self.guide_assignment is not None:
            blended_guides[self.guide_assignment != -1] = guides[
                self.guide_assignment[self.guide_assignment != -1]
            ]

        return blended_guides

    def compute_operator_guides(
        self, roots: torch.Tensor, guides: torch.Tensor
    ) -> torch.Tensor:
        self.update_random_parameters()

        if self.operator_guide_assignment is not None:
            guide_idx = self.operator_guide_idxs[self.operator_guide_assignment]
        else:
            guide_roots = guides[
                self.operator_guide_idxs, 0
            ]  # (M, 3), self.roots (N, 3)
            guide_roots_uv = self.roots_to_uv(guide_roots)
            roots_uv = self.roots_to_uv(roots) + self.uv_rand_offset

            operator_guide_idx = knn(
                self.operator_hg.id,
                roots_uv,
                guide_roots_uv,
                1,
                self.operator_search_radius,
                False,
            )[0].squeeze()
            guide_idx = self.operator_guide_idxs[operator_guide_idx]

        return guides[guide_idx], guide_idx

    def forward(
        self,
        roots: torch.Tensor,
        guides: torch.Tensor,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (guides - guides[:, 0].unsqueeze(1)) + roots.unsqueeze(1)


def compute_guide_from_root_and_dirs(root: torch.Tensor, dirs: torch.Tensor):
    return torch.cat((root.unsqueeze(1), dirs), dim=1).cumsum(dim=1)
