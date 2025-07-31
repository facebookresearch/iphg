# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def compute_strand_segment_lengths(strands):
    assert strands.shape[2] == 3
    segments = strands[:, 1:] - strands[:, :-1]
    return torch.sqrt(((segments + 1e-8) ** 2).sum(dim=-1))


def compute_strand_lengths(strands):
    return torch.sum(compute_strand_segment_lengths(strands), dim=-1)


def resample_strands(strands, n_samples, starting_idx=None, ending_idx=None):
    x = torch.linspace(0, 1, n_samples, device=strands.device)

    if starting_idx is not None or ending_idx is not None:
        starting_idx = (
            starting_idx
            if starting_idx is not None
            else torch.zeros((1,), device=strands.device)
        )
        ending_idx = ending_idx if ending_idx is not None else strands.shape[1]
        k = torch.clamp(ending_idx - starting_idx - 1, min=1)
        p = x[None] * k[:, None] + starting_idx[:, None]
        p = p[..., None].broadcast_to((p.shape[0], p.shape[1], 3))
        low = torch.floor(p).long()
        high = torch.ceil(p).long()
        frac = p - low.float()
        return (1 - frac) * strands.gather(dim=1, index=low) + frac * strands.gather(
            dim=1, index=high
        )

    k = strands.shape[1] - 1
    p = x * k
    low = torch.floor(p).long()
    high = torch.ceil(p).long()
    frac = (p - low.float())[None, :, None]
    return (1 - frac) * strands[:, low] + frac * strands[:, high]
