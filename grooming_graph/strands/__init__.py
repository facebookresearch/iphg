# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict
from scipy.spatial import Delaunay
from .geometry import repeat_edge_structure
from .preconditioner import PreconditionedParams

from grooming_graph.utils.math import dir_to_sph, sph_to_dir


class StrandParameterization:
    VERTEX = 0
    DIRECTION = 1
    CURVATURE = 2
    CURVATURE_LP = 3


def integrate(x: torch.Tensor, d: torch.Tensor):
    return torch.cat((x.unsqueeze(1), d), dim=1).cumsum(dim=1)


class Strands:
    def __init__(
        self,
        parameterization: StrandParameterization,
        strands: torch.Tensor,
        strand_root_uvs: torch.Tensor,
        intra_weight: float = 0.0,
        inter_weight: float = 1.0,
    ):
        self.parameterization = parameterization

        root_params = {}
        # Compute parameterization
        match parameterization:
            case StrandParameterization.DIRECTION:
                strand_dir = strands[:, 1:] - strands[:, :-1]
                params = {"dir": strand_dir}
                n_elem = strands.shape[1] - 1
            case StrandParameterization.VERTEX:
                strand_vertex = strands[:, 1:].clone()
                params = {"vertex": strand_vertex}
                n_elem = strands.shape[1] - 1
            case StrandParameterization.CURVATURE:
                strand_dir = strands[:, 1:] - strands[:, :-1]
                strand_curv = strand_dir[:, 1:] - strand_dir[:, :-1]
                params = {"curv": strand_curv}
                root_params = {"root_dirs": strands[:, 1] - strands[:, 0]}
                n_elem = strands.shape[1] - 2
            case StrandParameterization.CURVATURE_LP:
                strand_dir = strands[:, 1:] - strands[:, :-1]
                strand_length = torch.norm(strand_dir, dim=-1, keepdim=True) + 1e-12
                strand_dir_normalized = strand_dir / strand_length
                strand_angle = dir_to_sph(strand_dir_normalized)
                strand_curv = strand_angle[:, 1:] - strand_angle[:, :-1]
                strand_curv /= (strand_length[:, 1:] + strand_length[:, :-1]) / 2
                strand_curv = torch.cat(
                    (
                        torch.zeros_like(strand_curv[:, :1]),
                        strand_curv,
                    ),
                    dim=1,
                )
                params = {"curv": strand_curv, "length": strand_length}
                root_params = {
                    "root_dirs": strand_angle[:, 0],
                }
                n_elem = strands.shape[1] - 1
            case _:
                assert False, "Invalid strand parameterization!"

        n_strands = strands.shape[0]

        self._no_intra = intra_weight == 0.0

        if self._no_intra:
            intra_strand_edges = None
        else:
            # Compute edges for each strand
            intra_strand_edges = repeat_edge_structure(
                torch.stack(
                    (
                        torch.arange(n_elem - 1, device="cuda"),
                        torch.arange(1, n_elem, device="cuda"),
                    ),
                    dim=1,
                ),
                n_elem,
                n_strands,
            )

        # Compute edges between strands
        tri = Delaunay(strand_root_uvs.cpu())
        simplices = torch.tensor(tri.simplices, dtype=torch.long, device="cuda")

        root_edges = torch.cat(
            (simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]), dim=0
        )

        inter_strand_edges = repeat_edge_structure(n_elem * root_edges, 1, n_elem)

        # Create preconditioned params object
        self.pp = PreconditionedParams(
            params,
            intra_strand_edges,
            inter_strand_edges,
            intra_weight,
            inter_weight,
        )
        if root_params:
            self.root_pp = PreconditionedParams(
                root_params,
                None,
                root_edges,
                0,
                inter_weight,
            )
        else:
            self.root_pp = None

    @property
    def params(self):
        u = self.pp.params.copy()
        if self.root_pp is not None:
            u = {**u, **self.root_pp.params}
        return u

    @property
    def _unflattened_params(self) -> Dict[str, torch.Tensor]:
        param_dict = self.pp._unflattened_params.copy()
        if self.root_pp is not None:
            param_dict = {**param_dict, **self.root_pp._unflattened_params}
        return param_dict

    @_unflattened_params.setter
    def _unflattened_params(self, param_dict: Dict[str, torch.Tensor]):
        param_dict = param_dict.copy()
        root_param_dict = {}
        for k in list(param_dict.keys()):
            if k.startswith("root") and self.root_pp is not None:
                root_param_dict[k] = param_dict.pop(k)
        self.pp._unflattened_params = param_dict
        if self.root_pp is not None:
            self.root_pp._unflattened_params = root_param_dict

    def compute_strands(self, roots: torch.Tensor) -> torch.Tensor:
        match self.parameterization:
            case StrandParameterization.DIRECTION:
                dirs = self._unflattened_params["dir"]
                return integrate(roots, dirs)
            case StrandParameterization.VERTEX:
                strand_vertices = self._unflattened_params["vertex"]
                return torch.cat((roots.unsqueeze(1), strand_vertices), dim=1)
            case StrandParameterization.CURVATURE:
                params = self._unflattened_params
                root_dirs, curv = params["root_dirs"], params["curv"]
                dirs = integrate(root_dirs, curv)
                return integrate(roots, dirs)
            case StrandParameterization.CURVATURE_LP:
                params = self._unflattened_params
                root_dirs = params["root_dirs"]
                length = params["length"]
                curv = params["curv"][:, 1:]
                dirs = (
                    sph_to_dir(
                        integrate(
                            root_dirs, curv * (length[:, 1:] + length[:, :-1]) / 2
                        )
                    )
                    * length
                )
                return integrate(roots, dirs)
