# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def sph_to_dir(angles):
    theta = angles[..., 0]
    phi = angles[..., 1]
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    return torch.stack((cp * st, sp * st, ct), dim=2)


def dir_to_sph(dir):
    theta = torch.acos(dir[..., 2])
    phi = torch.atan2(dir[..., 1], dir[..., 0])
    return torch.stack((theta, phi), dim=2)
