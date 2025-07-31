# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import warp as wp
from typing import Callable, Tuple


wp.init()


@wp.kernel
def point_mesh_closest_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    closest_faces: wp.array(dtype=wp.int32),
    closest_points: wp.array(dtype=wp.vec3),
    bary_coords: wp.array(dtype=wp.vec3),
    dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]

    mqp = wp.mesh_query_point_sign_normal(mesh_id, p, 1e9)
    closest_p = wp.mesh_eval_position(mesh_id, mqp.face, mqp.u, mqp.v)

    closest_faces[tid] = mqp.face
    closest_points[tid] = closest_p
    bary_coords[tid] = wp.vec3(
        mqp.u,
        mqp.v,
        1.0 - mqp.u - mqp.v,
    )
    dists[tid] = mqp.sign * wp.length(closest_p - p)


class PointMeshClosestPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mesh_id, points):
        ctx.tape = wp.Tape()
        ctx.points = wp.from_torch(points, dtype=wp.vec3, requires_grad=False)
        ctx.points.requires_grad = points.requires_grad

        faces = wp.zeros(len(points), device=ctx.points.device, dtype=wp.int32)
        ctx.closest_points = wp.zeros(
            len(points),
            dtype=wp.vec3,
            device=ctx.points.device,
            requires_grad=ctx.points.requires_grad,
        )
        ctx.bary_coords = wp.zeros(
            len(points),
            dtype=wp.vec3,
            device=ctx.points.device,
            requires_grad=ctx.points.requires_grad,
        )
        ctx.dists = wp.zeros(
            len(points),
            device=ctx.points.device,
            requires_grad=ctx.points.requires_grad,
        )

        with ctx.tape:
            wp.launch(
                point_mesh_closest_point_kernel,
                dim=len(points),
                inputs=[mesh_id, ctx.points],
                outputs=[faces, ctx.closest_points, ctx.bary_coords, ctx.dists],
            )

        return (
            wp.to_torch(faces),
            wp.to_torch(ctx.closest_points),
            wp.to_torch(ctx.bary_coords),
            wp.to_torch(ctx.dists),
        )

    @staticmethod
    def backward(ctx, _, closest_points_grad, bary_coords_grad, dists_grad):
        ctx.closest_points.grad = wp.from_torch(closest_points_grad, dtype=wp.vec3)
        ctx.bary_coords.grad = wp.from_torch(bary_coords_grad, dtype=wp.vec3)
        ctx.dists.grad = wp.from_torch(dists_grad, dtype=wp.float32)
        ctx.tape.backward()

        return None, wp.to_torch(ctx.tape.gradients[ctx.points])


point_mesh_closest_point: Callable[
    [int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = PointMeshClosestPoint.apply


@wp.kernel
def point_mesh_dist_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = points[tid]

    mqp = wp.mesh_query_point_sign_normal(mesh_id, p, 1e9)
    closest_p = wp.mesh_eval_position(mesh_id, mqp.face, mqp.u, mqp.v)

    dists[tid] = mqp.sign * wp.length(closest_p - p)


class PointMeshDist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mesh_id, points):
        ctx.tape = wp.Tape()
        ctx.points = wp.from_torch(points, dtype=wp.vec3, requires_grad=False)
        ctx.points.requires_grad = points.requires_grad
        ctx.dists = wp.zeros(
            len(points),
            device=ctx.points.device,
            requires_grad=ctx.points.requires_grad,
        )

        with ctx.tape:
            wp.launch(
                point_mesh_dist_kernel,
                dim=len(points),
                inputs=[mesh_id, ctx.points],
                outputs=[ctx.dists],
            )

        return wp.to_torch(ctx.dists)

    @staticmethod
    def backward(ctx, dists_grad):
        ctx.dists.grad = wp.from_torch(dists_grad, dtype=wp.float32)
        ctx.tape.backward()

        return None, wp.to_torch(ctx.tape.gradients[ctx.points])


point_mesh_dist: Callable[[int, torch.Tensor], torch.Tensor] = PointMeshDist.apply


@wp.kernel
def point_grid_closest_point_kernel(
    grid: wp.uint64,
    radius: float,
    ignore_self: bool,
    query_points: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # query point
    p = query_points[tid]

    # create grid query around point
    query = wp.hash_grid_query(grid, p, radius)
    index = int(0)

    closest_point = wp.vec3()
    closest_dist = float(1e99)

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        skip = ignore_self and tid == index

        if not skip:
            dist = wp.length(p - neighbor)
            if dist < closest_dist:
                closest_point = neighbor
                closest_dist = dist

    output[tid] = closest_point


def point_grid_closest_point_index(
    grid_id: int,
    query_points: torch.Tensor,
    points: wp.array,
    radius: float,
    ignore_self: bool = False,
) -> torch.Tensor:
    query_points = wp.from_torch(query_points, dtype=wp.vec3)
    output = wp.zeros(len(query_points), dtype=wp.vec3, device=query_points.device)

    wp.launch(
        point_grid_closest_point_kernel,
        dim=len(query_points),
        inputs=[grid_id, radius, ignore_self, query_points, points],
        outputs=[output],
    )

    return wp.to_torch(output)


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/structures/meshes.py
def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    verts_normals = torch.zeros_like(verts)
    vertices_faces = verts[faces]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    # NOTE: this is already applying the area weighting as the magnitude
    # of the cross product is 2 x area of the triangle.
    verts_normals = verts_normals.index_add(0, faces[:, 0], faces_normals)
    verts_normals = verts_normals.index_add(0, faces[:, 1], faces_normals)
    verts_normals = verts_normals.index_add(0, faces[:, 2], faces_normals)

    return torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)
