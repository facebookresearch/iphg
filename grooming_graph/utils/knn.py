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
def knn_kernel(
    grid: wp.uint64,
    k: int,
    radius: float,
    ignore_self: bool,
    query_points: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    closest_idxs: wp.array2d(dtype=wp.int32),
    dists: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()

    # query point
    p = query_points[tid]

    # create grid query around point
    query = wp.hash_grid_query(grid, p, radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        skip = ignore_self and tid == index

        if not skip:
            dist = wp.length(p - neighbor)
            for i in range(k):
                if dist < dists[tid, i]:
                    for j in range(k - 1, i, -1):
                        dists[tid, j] = dists[tid, j - 1]
                        closest_idxs[tid, j] = closest_idxs[tid, j - 1]
                    dists[tid, i] = dist
                    closest_idxs[tid, i] = index
                    break


class KNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid_id, query_points, points, k, radius, ignore_self):
        ctx.dim = query_points.shape[1]
        assert ctx.dim == points.shape[1] and ctx.dim in [2, 3]
        if ctx.dim == 2:
            query_points = torch.cat(
                (query_points, torch.zeros_like(query_points[:, 0:1])), dim=1
            )
            points = torch.cat((points, torch.zeros_like(points[:, 0:1])), dim=1)

        ctx.tape = wp.Tape()
        ctx.query_points = wp.from_torch(
            query_points, dtype=wp.vec3, requires_grad=False
        )
        ctx.query_points.requires_grad = query_points.requires_grad
        ctx.points = wp.from_torch(points, dtype=wp.vec3, requires_grad=False)
        ctx.points.requires_grad = points.requires_grad
        ctx.closest_idxs = wp.zeros(
            (len(query_points), k), dtype=wp.int32, device=ctx.query_points.device
        )
        ctx.dists = wp.full(
            (len(query_points), k),
            1e99,
            dtype=wp.float32,
            device=ctx.query_points.device,
        )

        with ctx.tape:
            wp.launch(
                knn_kernel,
                dim=len(query_points),
                inputs=[grid_id, k, radius, ignore_self, ctx.query_points, ctx.points],
                outputs=[ctx.closest_idxs, ctx.dists],
            )

        return wp.to_torch(ctx.closest_idxs), wp.to_torch(ctx.dists)

    @staticmethod
    def backward(ctx, _, dists_grad):
        ctx.dists.grad = wp.from_torch(dists_grad, dtype=wp.float32)
        ctx.tape.backward()

        query_points_grad = (
            wp.to_torch(ctx.tape.gradients[ctx.query_points])
            if ctx.query_points.requires_grad
            else None
        )
        points_grad = (
            wp.to_torch(ctx.tape.gradients[ctx.points])
            if ctx.points.requires_grad
            else None
        )
        if ctx.dim == 2:
            if query_points_grad is not None:
                query_points_grad = query_points_grad[:, :2]
            if points_grad is not None:
                points_grad = points_grad[:, :2]

        return (
            None,
            query_points_grad,
            points_grad,
            None,
            None,
            None,
        )


knn: Callable[
    [int, torch.Tensor, torch.Tensor, int, float, bool],
    Tuple[torch.Tensor, torch.Tensor],
] = KNN.apply
