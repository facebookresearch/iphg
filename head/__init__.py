# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import nvdiffrast.torch as dr
import warp as wp
import meshio
import numpy as np
from torchvision.io.image import read_image
from typing import Sequence, Optional
from scipy.stats import qmc
from head.geometry_utils import (
    compute_vertex_normals,
    point_mesh_closest_point,
    point_mesh_dist,
    point_grid_closest_point_index,
)


class HeadModel:
    def __init__(
        self,
        path: str,
        transform: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.glctx = dr.RasterizeCudaContext()

        m = meshio.read(os.path.join(path, "head_flame.obj"))
        self.vert = torch.tensor(m.points, dtype=torch.float32, device="cuda")
        self.tri = torch.tensor(m.cells[0].data, dtype=torch.int32, device="cuda")
        self.normals = compute_vertex_normals(self.vert, self.tri)

        if transform is not None:
            vert = torch.cat((self.vert, torch.ones_like(self.vert[:, 0:1])), dim=1)
            mtx = torch.tensor(transform, dtype=torch.float32, device="cuda")
            self.vert = (vert @ mtx.t())[:, :3].contiguous()

        bb_min = self.vert.min(dim=0).values
        bb_max = self.vert.max(dim=0).values
        center = (bb_min + bb_max) / 2
        self.extent_length = torch.norm(bb_max - bb_min)
        scale = 4.4597 / self.extent_length
        self.transform = torch.tensor(
            [
                [scale, 0, 0, -center[0] * scale],
                [0, scale, 0, -center[1] * scale],
                [0, 0, scale, -center[2] * scale],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        self.vert = self.scale_to_head(self.vert).contiguous()

        # Load precomputed scalp data
        cur_path = os.path.abspath(os.path.dirname(__file__))
        # Local scalp data
        if os.path.exists(os.path.join(path, "new_scalp_vertex_idx.npy")):
            head_path = path
        else:
            head_path = os.path.join(cur_path, "..", "data/head")

        scalp_vert_idx = torch.tensor(
            np.load(os.path.join(head_path, "new_scalp_vertex_idx.npy")),
            dtype=torch.long,
            device="cuda",
        )
        self.scalp_faces = torch.tensor(
            np.load(os.path.join(head_path, "new_scalp_faces.npy")),
            dtype=torch.int32,
            device="cuda",
        )
        self.scalp_verts = self.vert[scalp_vert_idx]
        self.scalp_normals = compute_vertex_normals(self.scalp_verts, self.scalp_faces)

        scalp_uvs = torch.tensor(
            np.load(os.path.join(head_path, "new_scalp_uvcoords.npy")),
            dtype=torch.float32,
            device="cuda",
        )
        self.scalp_pos_to_uv = scalp_uvs
        self.uv_map_dim = 1000
        self.boundary_search_radius = (
            0.7071  # sqrt(0.5), half the diagonal of the texture map
        )
        # Precompute inverse uv map
        (
            self.scalp_uv_to_pos,
            self.scalp_uv_to_pos_mask,
            self.scalp_uv_to_pos_extended_mask,
            scalp_boundary_uv,
        ) = self._compute_inverse_uv_mapping(
            self.scalp_verts, self.scalp_faces, scalp_uvs, self.uv_map_dim
        )

        boundary_points = torch.cat(
            (scalp_boundary_uv, torch.zeros_like(scalp_boundary_uv[:, 0:1])),
            dim=1,
        )
        self.boundary_points = wp.from_torch(boundary_points, dtype=wp.vec3)
        self.boundary_hg = wp.HashGrid(
            128,
            128,
            1,
            device="cuda",
        )
        self.boundary_hg.build(self.boundary_points, self.boundary_search_radius)

        self.scalp_mesh = wp.Mesh(
            wp.from_torch(self.scalp_verts, dtype=wp.vec3),
            wp.from_torch(self.scalp_faces.flatten(), dtype=wp.int32),
        )

        scalp_mask_path = os.path.join(path, "scalp_mask.png")
        if os.path.exists(scalp_mask_path):
            scalp_mask_r = read_image(scalp_mask_path)[0].cuda()
            self.scalp_mask = scalp_mask_r > 0
        else:  # Default scalp mask
            scalp_mask_r = read_image(os.path.join(head_path, "scalp_mask.png"))[
                0
            ].cuda()
            self.scalp_mask = scalp_mask_r > 0
        self.uv_sampler = qmc.Halton(2, optimization="lloyd", seed=0)

        # Remove ears from head when using for collision
        if os.path.exists(os.path.join(head_path, "ears_vertex_idx.npy")):
            ears_idx_set = set(np.load(os.path.join(head_path, "ears_vertex_idx.npy")))
            collidable_tri = []
            for t in m.cells[0].data.tolist():
                if all(v not in ears_idx_set for v in t):
                    collidable_tri.append(t)
            collidable_tri = torch.tensor(
                collidable_tri, dtype=torch.int32, device="cuda"
            )
        else:
            collidable_tri = self.tri

        self.head_normals = compute_vertex_normals(self.vert, collidable_tri)
        self.head_mesh = wp.Mesh(
            wp.from_torch(self.vert, dtype=wp.vec3),
            wp.from_torch(collidable_tri.flatten(), dtype=wp.int32),
        )

    def scale_to_head(self, points: torch.Tensor) -> torch.Tensor:
        shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.cat((points, torch.ones_like(points[:, 0:1])), dim=1)
        points = (points @ self.transform.t())[:, :3]
        return points.reshape(shape)

    def scale_from_head(self, points: torch.Tensor) -> torch.Tensor:
        shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.cat((points, torch.ones_like(points[:, 0:1])), dim=1)
        points = (points @ self.transform.t().inverse())[:, :3]
        return points.reshape(shape)

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def compute_head_color(self, cam_center: torch.Tensor) -> torch.Tensor:
        view_dir = torch.nn.functional.normalize(
            self.scale_from_head(self.vert) - cam_center[None, :], dim=-1
        )
        return (-view_dir * self.normals).sum(dim=-1, keepdim=True).abs().repeat(1, 3)

    def _compute_inverse_uv_mapping(self, verts, tris, uvs, uv_map_dim=1000):
        uvs_clip = 2 * uvs - 1
        uvs_clip = torch.concat((uvs_clip, torch.ones_like(uvs_clip)), dim=1)[None]
        tris = tris.int().contiguous()

        rast, _ = dr.rasterize(
            self.glctx,
            uvs_clip,
            tris,
            resolution=[uv_map_dim, uv_map_dim],
        )
        pos, _ = dr.interpolate(verts.unsqueeze(0), rast, tris)
        # Texels that are empty are marked with a mask of 0
        mask = rast[..., 3] > 0

        pos = pos.permute((0, 2, 1, 3)).squeeze(0)
        mask = mask.permute((0, 2, 1)).squeeze(0)

        # Find the texels on the boundary of the uv map
        # A texel is on the boundary if it has a mask of 1 AND
        # at least one of its neighbors has a mask of 0 OR
        # it is at the edge of the uv map
        neighbor_kernel = torch.ones((1, 1, 3, 3), device="cuda")
        boundary = mask & (
            torch.nn.functional.conv2d(
                mask[None, None].float(), neighbor_kernel, padding=1
            )
            < 9
        ).reshape(mask.shape)
        boundary_uv = torch.nonzero(boundary)
        boundary_uv = boundary_uv.float() / torch.tensor(
            [uv_map_dim - 1, uv_map_dim - 1], device="cuda"
        )

        # Extend the uv map to a few texels outside the boundary to prevent
        # sampling outside the uv map due to numerical errors
        n_texels = 10
        neighbor_kernel = torch.ones(
            (1, 1, 2 * n_texels + 1, 2 * n_texels + 1), device="cuda"
        )
        n_nz_neighbors = (
            torch.nn.functional.conv2d(
                mask[None, None].float(), neighbor_kernel, padding=n_texels
            )
            .squeeze(0)
            .squeeze(0)
        )
        # The extended boundary is the texels that are outside the boundary
        # These are the ones that have a mask of 0 and at least one non-zero neighbor
        extended_boundary = ~mask & (n_nz_neighbors > 0).reshape(mask.shape)

        # Extend the uv map by averaging the values of the non-zero neighbors
        pos += torch.where(
            extended_boundary,
            torch.nn.functional.conv2d(
                pos.permute(2, 0, 1).unsqueeze(1), neighbor_kernel, padding=n_texels
            ).squeeze(1)
            / n_nz_neighbors.unsqueeze(0),
            0,
        ).permute(1, 2, 0)

        extended_mask = mask | extended_boundary

        return pos, mask, extended_mask, boundary_uv

    def project_strands_onto_scalp(self, strands: torch.Tensor) -> torch.Tensor:
        roots_proj = point_mesh_closest_point(self.scalp_mesh.id, strands[:, 0])[
            1
        ].unsqueeze(1)
        return torch.concat((roots_proj, strands[:, 1:]), dim=1)

    def check_points_on_scalp(self, points: torch.Tensor, eps=5e-3) -> bool:
        closest_point = point_mesh_closest_point(self.scalp_mesh.id, points)[1]

        return torch.all((points - closest_point).norm(dim=1) < eps)

    def scalp_get_normal(self, points: torch.Tensor) -> torch.Tensor:
        face_idxs, _, bary, _ = point_mesh_closest_point(self.scalp_mesh.id, points)
        faces = self.scalp_faces[face_idxs]
        bary = torch.clamp(bary, 0, 1)

        vert_normals = self.scalp_normals[faces]
        normals = torch.nn.functional.normalize(
            bary[:, 0, None] * vert_normals[:, 0]
            + bary[:, 1, None] * vert_normals[:, 1]
            + bary[:, 2, None] * vert_normals[:, 2]
        )
        return normals

    def scalp_uv_mapping(self, points: torch.Tensor) -> torch.Tensor:
        face_idxs, _, bary, _ = point_mesh_closest_point(self.scalp_mesh.id, points)
        faces = self.scalp_faces[face_idxs]
        bary = torch.clamp(bary, 0, 1)

        vert_uvs = self.scalp_pos_to_uv[faces]
        uvs = (
            bary[:, 0, None] * vert_uvs[:, 0]
            + bary[:, 1, None] * vert_uvs[:, 1]
            + bary[:, 2, None] * vert_uvs[:, 2]
        )
        return uvs

    def project_uvs_onto_scalp(self, uvs: torch.Tensor) -> torch.Tensor:
        points = torch.cat((uvs, torch.zeros_like(uvs[:, 0:1])), dim=1)
        closest_points = point_grid_closest_point_index(
            self.boundary_hg.id,
            points,
            self.boundary_points,
            self.boundary_search_radius,
        )
        boundary_uv = closest_points[:, :2]

        th, tw, _ = self.scalp_uv_to_pos.shape
        posf = uvs * torch.tensor([th - 1, tw - 1], device="cuda") - 0.5
        posi = posf.floor().long()
        posi = torch.clamp(posi, min=0)

        yi, xi = posi[:, 0], posi[:, 1]
        mask = self.scalp_uv_to_pos_mask[yi, xi, None].broadcast_to(uvs.shape)

        # Only return the projected UVs if they are outside the boundary
        return torch.where(mask, uvs, boundary_uv)

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def _scalp_inverse_uv_mapping_helper(self, uvs: torch.Tensor) -> torch.Tensor:
        th, tw, _ = self.scalp_uv_to_pos.shape

        posf = uvs * torch.tensor([th - 1, tw - 1], device="cuda") - 0.5
        posi = posf.floor().long()
        posf = posf - posi.float()
        posi = torch.clamp(posi, min=0)

        yi, xi = posi[:, 0], posi[:, 1]
        yf, xf = posf[:, 0], posf[:, 1]

        # The mask is 0 if no UV info is available
        w00 = (1 - yf) * (1 - xf) * self.scalp_uv_to_pos_extended_mask[yi, xi]
        w01 = (1 - yf) * xf * self.scalp_uv_to_pos_extended_mask[yi, xi + 1]
        w10 = yf * (1 - xf) * self.scalp_uv_to_pos_extended_mask[yi + 1, xi]
        w11 = yf * xf * self.scalp_uv_to_pos_extended_mask[yi + 1, xi + 1]
        w_sum = w00 + w01 + w10 + w11
        w_sum = w_sum[:, None]

        t00 = self.scalp_uv_to_pos[yi, xi] * w00[:, None]
        t01 = self.scalp_uv_to_pos[yi, xi + 1] * w01[:, None]
        t10 = self.scalp_uv_to_pos[yi + 1, xi] * w10[:, None]
        t11 = self.scalp_uv_to_pos[yi + 1, xi + 1] * w11[:, None]
        t_sum = t00 + t01 + t10 + t11

        return t_sum, w_sum

    def scalp_inverse_uv_mapping(self, uvs: torch.Tensor) -> torch.Tensor:
        t_sum, w_sum = self._scalp_inverse_uv_mapping_helper(uvs)

        assert torch.all(w_sum > 0)
        return t_sum / w_sum

    def scalp_sample_uvs(self, n_samples: int) -> torch.Tensor:
        mask: torch.Tensor = self.scalp_uv_to_pos_mask & self.scalp_mask
        mask_nz = mask.nonzero()
        mask_bound_min = mask_nz.min(dim=0).values
        mask_bound_max = mask_nz.max(dim=0).values
        mask_bounded = mask[
            mask_bound_min[0] : mask_bound_max[0] + 1,
            mask_bound_min[1] : mask_bound_max[1] + 1,
        ]
        masked_area_ratio = mask_bounded.sum().float() / mask_bounded.numel()
        n = int(n_samples / masked_area_ratio)

        uvs = torch.empty((0, 2), device="cuda")
        th, tw, _ = self.scalp_uv_to_pos.shape
        scale = torch.tensor([th - 1, tw - 1], device="cuda")
        uv_min = mask_bound_min.float() / scale
        uv_max = mask_bound_max.float() / scale

        while uvs.shape[0] < n_samples:
            new_uvs = torch.tensor(
                self.uv_sampler.random(n), dtype=torch.float32, device="cuda"
            )
            new_uvs = new_uvs * (uv_max - uv_min) + uv_min

            posi = (new_uvs * scale).floor().long()
            yi, xi = posi[:, 0], posi[:, 1]
            new_uvs = new_uvs[mask[yi, xi]]
            uvs = torch.cat((uvs, new_uvs))

        return uvs[:n_samples]

    def project_points_outside_head(
        self, points: torch.Tensor, eps=1e-2
    ) -> torch.Tensor:
        face_idxs, closest_point, bary, dists = point_mesh_closest_point(
            self.head_mesh.id, points
        )

        faces = self.tri[face_idxs]
        bary = torch.clamp(bary, 0, 1)
        vert_normals = self.head_normals[faces]
        normals = torch.nn.functional.normalize(
            bary[:, 0, None] * vert_normals[:, 0]
            + bary[:, 1, None] * vert_normals[:, 1]
            + bary[:, 2, None] * vert_normals[:, 2]
        )
        new_points = points.clone()
        new_points[dists <= 0] = closest_point[dists <= 0] + eps * normals[dists <= 0]

        return new_points

    def compute_distance_to_head(self, points: torch.Tensor) -> torch.Tensor:
        return point_mesh_dist(self.head_mesh.id, points)

    def resample_roots(
        self, roots: torch.Tensor, n_roots: int, eps_scale=8
    ) -> torch.Tensor:
        roots_uv = self.scalp_uv_mapping(roots)

        # Build a hash grid for root UVS to compute distances
        search_radius = 0.7071
        roots_points = torch.cat(
            (roots_uv, torch.zeros_like(roots_uv[:, 0:1])),
            dim=1,
        )
        roots_points = wp.from_torch(roots_points, dtype=wp.vec3)
        roots_hg = wp.HashGrid(
            128,
            128,
            1,
            device="cuda",
        )
        roots_hg.build(roots_points, search_radius)

        def distance_to_roots(uvs, ignore_self=False):
            points = torch.cat((uvs, torch.zeros_like(uvs[:, 0:1])), dim=1)
            closest_points = point_grid_closest_point_index(
                roots_hg.id,
                points,
                roots_points,
                search_radius,
                ignore_self=ignore_self,
            )
            return (uvs - closest_points[:, :2]).norm(dim=1)

        eps = distance_to_roots(roots_uv, ignore_self=True).mean() * eps_scale

        # Compute estimate of the number of samples required accounting for rejection
        th, tw, _ = self.scalp_uv_to_pos.shape
        scale = torch.tensor([th - 1, tw - 1], device="cuda")

        roots_uv_bound_min = roots_uv.min(dim=0).values
        roots_uv_bound_max = roots_uv.max(dim=0).values
        roots_uv_extent = roots_uv_bound_max - roots_uv_bound_min
        n = int(n_roots / (roots_uv_extent[0] * roots_uv_extent[1]))

        uvs = torch.empty((0, 2), device="cuda")

        uv_min = roots_uv_bound_min
        uv_max = roots_uv_bound_max
        mask: torch.Tensor = self.scalp_uv_to_pos_mask

        sampler = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=0)

        while uvs.shape[0] < n_roots:
            new_uvs = sampler.draw(n).cuda()
            new_uvs = new_uvs * (uv_max - uv_min) + uv_min

            dist = distance_to_roots(new_uvs)
            new_uvs = new_uvs[dist < eps]

            posi = (new_uvs * scale).floor().long()
            yi, xi = posi[:, 0], posi[:, 1]
            new_uvs = new_uvs[mask[yi, xi]]

            uvs = torch.cat((uvs, new_uvs))

        return self.scalp_inverse_uv_mapping(uvs[:n_roots])
