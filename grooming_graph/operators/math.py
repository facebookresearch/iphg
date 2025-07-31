# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    device = axis.device
    shape = axis.shape if (len(angle.shape) < 2 or axis.shape[1] > angle.shape[1]) else angle.shape
    rotation = torch.zeros(shape[:-1] + (3, 3), device=device)

    s = torch.sin(angle.squeeze(-1))
    c = torch.cos(angle.squeeze(-1))

    t = 1.0 - c
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    rotation[..., 0, 0] = t * x * x + c
    rotation[..., 1, 0] = t * x * y - s * z
    rotation[..., 2, 0] = t * x * z + s * y
    rotation[..., 0, 1] = t * x * y + s * z
    rotation[..., 1, 1] = t * y * y + c
    rotation[..., 2, 1] = t * y * z - s * x
    rotation[..., 0, 2] = t * x * z - s * y
    rotation[..., 1, 2] = t * y * z + s * x
    rotation[..., 2, 2] = t * z * z + c

    return rotation
