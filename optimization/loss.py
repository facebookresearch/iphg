# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from sklearn.neighbors import KDTree
from head import HeadModel


def loss_swd(x: torch.Tensor, y: torch.Tensor):
    fx = x.reshape(1, x.shape[0], -1)
    fy = y.reshape(1, y.shape[0], -1)
    dim = fx.shape[2]

    # sample random directions
    Mdirection = dim
    directions = torch.randn(dim, Mdirection, device=x.device)
    directions = directions / torch.sqrt(torch.sum(directions**2, dim=0, keepdim=True))
    # project features over random directions
    projected_fx = torch.einsum("bnd,dm->bnm", fx, directions)
    projected_fy = torch.einsum("bnd,dm->bnm", fy, directions)
    # sort the projections
    sorted_fx = torch.sort(projected_fx, dim=1)[0]
    sorted_fy = torch.sort(projected_fy, dim=1)[0]
    # L2 over sorted lists
    loss = torch.mean((sorted_fx - sorted_fy).square())

    return loss


def rand_loss(
    x: torch.Tensor, y: torch.Tensor, lambda_d=1.0, lambda_l=1e-2
) -> torch.Tensor:
    def compute_features(x):
        @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
        def compute_x_dir_length(x):
            x_seg = x[:, 1:] - x[:, :-1]
            x_length = torch.norm(x_seg, dim=-1, keepdim=True) + 1e-12
            x_dir = x_seg / x_length
            return x_dir, x_length

        x_dir, x_length = compute_x_dir_length(x)
        xf = torch.fft.rfft(x_dir, dim=1).abs()
        return x_dir, xf, x_length

    x_dir, xf, x_length = compute_features(x)
    y_dir, yf, y_length = compute_features(y)
    return (
        (xf - yf).square().mean()
        + lambda_d * (1 - (x_dir * y_dir).sum(-1)).mean()
        + lambda_l * (x_length - y_length).square().mean()
    )


def shape_loss(x: torch.Tensor, y: torch.Tensor, lambda_e=4e2) -> torch.Tensor:
    def compute_features(x):
        x_seg = x[:, 1:] - x[:, :-1]
        xf = torch.fft.rfft(x_seg, dim=1).abs()
        return x_seg, xf

    x_seg, xf = compute_features(x)
    y_seg, yf = compute_features(y)

    return loss_swd(xf, yf) + lambda_e * loss_swd(x_seg, y_seg)


class RandomStrandSelector:
    def __init__(self, head: HeadModel, roots: torch.Tensor):
        self.head = head
        self.roots_uv = head.scalp_uv_mapping(roots).cpu().numpy()
        self.tree = KDTree(self.roots_uv)

    def select(self, n_batches: int, n_samples: int) -> torch.Tensor:
        rand_idx = np.random.choice(len(self.roots_uv), size=n_batches)
        return self.tree.query(self.roots_uv[rand_idx], 1 + n_samples)[1][:, 1:]


def dpp_loss(rss: RandomStrandSelector, x, y, compute_pp_det, n_total_samples):
    n_samples = max(min(x.shape[0] // 100, 200), 1)
    n_batches = max(n_total_samples // n_samples, 1)

    def get_batches(x, idx):
        return x[idx].reshape(n_batches, n_samples, x.shape[-2], x.shape[-1])

    p_idx = rss.select(n_batches, n_samples).flatten()

    dx = compute_pp_det(get_batches(x, p_idx))
    dy = compute_pp_det(get_batches(y, p_idx))
    return (dx - dy).square().mean()


def scale_loss(
    rss: RandomStrandSelector,
    x: torch.Tensor,
    y: torch.Tensor,
    n_total_samples=2000,
    sigma=1e-1,
) -> torch.Tensor:
    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def compute_pp_det(x):
        x_seg = x[:, :, 1:] - x[:, :, :-1]
        x_length = torch.norm(x_seg, dim=-1).sum(-1)
        x_dev = (x_length[:, None] - x_length[:, :, None]).square()
        dists = torch.exp(-x_dev / sigma) + 1e-3 * torch.eye(
            x_dev.shape[1], device=x.device
        )
        return torch.linalg.slogdet(dists)[1]

    return dpp_loss(rss, x, y, compute_pp_det, n_total_samples)


def clump_loss(
    rss: RandomStrandSelector,
    x: torch.Tensor,
    y: torch.Tensor,
    n_total_samples=2000,
    sigma=1e-1,
) -> torch.Tensor:
    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def compute_pp_det(x):
        x_dev = (x[:, None] - x[:, :, None]).norm(dim=-1)
        x_dev = x_dev / (x_dev[..., 0:1] + 1e-8)
        x_dev = x_dev.mean(-1)
        dists = torch.exp(-x_dev / sigma) + 1e-3 * torch.eye(
            x_dev.shape[1], device=x.device
        )
        return torch.linalg.slogdet(dists)[1]

    return dpp_loss(rss, x, y, compute_pp_det, n_total_samples)


def hair_head_penetration_reg(
    head: HeadModel, x: torch.Tensor, threshold=0.05, lambda_=0.25
) -> torch.Tensor:
    hair_penetration_reg = torch.clamp(
        threshold - head.compute_distance_to_head(x[:, 1:].reshape(-1, 3)),
        min=0,
    )
    hair_penetration_reg = hair_penetration_reg.sum() / (
        hair_penetration_reg.count_nonzero() + 1e-5
    )
    return lambda_ * hair_penetration_reg
