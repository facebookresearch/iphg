# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict, Optional
from .geometry import compute_strand_matrix


class PreconditionedParams:
    def __init__(
        self,
        params: Dict[str, torch.Tensor],
        intra_edges: Optional[torch.Tensor],
        inter_edges: torch.Tensor,
        intra_weight: float = 1.0,
        inter_weight: float = 1.0,
    ):
        assert len(params) > 0, "At least one parameter must be provided"
        a_param = next(iter(params.values()))
        assert all(
            param.dim() >= 2 for param in params.values()
        ), "Parameters must be of shape (..., d)"
        assert all(
            param.shape[:-1] == a_param.shape[:-1] for param in params.values()
        ), "Parameters must have the same shape"

        self.use_preconditioner = intra_weight != 0.0 or inter_weight != 0.0
        self.params = params
        self.shape = a_param.shape[:-1]

        # Not using preconditioner, so we can directly optimize the parameters
        if not self.use_preconditioner:
            for param in self.params.values():
                param.requires_grad_(True)
            return

        # Compute preconditioned matrix
        self.M = compute_strand_matrix(
            a_param.shape[:-1].numel(),
            intra_edges,
            inter_edges,
            intra_weight,
            inter_weight,
        )

        # Flatten to (n, d) for optimization
        for k, param in self.params.items():
            self.params[k] = self.flatten(param).requires_grad_(True)

    def flatten(self, p: torch.Tensor) -> torch.Tensor:
        if not self.use_preconditioner:
            return p

        return p.reshape(-1, p.shape[-1])

    def unflatten(self, p: torch.Tensor) -> torch.Tensor:
        if not self.use_preconditioner:
            return p

        return p.reshape(self.shape + (p.shape[-1],))

    @property
    def _unflattened_params(self):
        ds = [param.shape[-1] for param in self.params.values()]
        p = self.unflatten(torch.cat(list(self.params.values()), dim=-1))
        ps = torch.split(p, ds, dim=-1)

        return {k: ps[i] for i, k in enumerate(self.params.keys())}

    @_unflattened_params.setter
    def _unflattened_params(self, param_dict: Dict[str, torch.Tensor]):
        assert len(param_dict) == len(self.params), "Number of parameters must match"

        ds = [param.shape[-1] for param in self.params.values()]
        u = self.flatten(torch.cat(list(param_dict.values()), dim=-1))
        us = torch.split(u, ds, dim=-1)

        for i, k in enumerate(param_dict.keys()):
            self.params[k][...] = us[i]
