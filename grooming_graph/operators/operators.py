# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Optional
from .operator import Operator, BoundedParameter, RandomParameter, RandMode
from .noise import fractal_noise3
from .math import axis_angle_rotation

from grooming_graph.utils.hair import compute_strand_lengths


class Scale(Operator):
    def __init__(
        self,
        scale: float = 1.0,
        rand_scale: Optional[float] = None,
        optimizable: bool = True,
        rand_mode: RandMode = RandMode.FIX,
        n_strands: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(rand_mode, generator)
        self.scale = BoundedParameter(
            scale,
            min_value=1e-2,
            requires_grad=False,
        )
        self.rand_scale = BoundedParameter(
            rand_scale,
            default_value=0.0,
            min_value=0.0,
            requires_grad=optimizable,
        )

        self.rand = RandomParameter(
            (n_strands, 1, 1),
            distribution="normal",
            requires_grad=rand_mode == RandMode.OPTIMIZE and optimizable,
        )

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def forward(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = self.rand * self.rand_scale + self.scale

        return strands[:, :1] + scale * (strands - strands[:, :1])

    def mean(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return strands


class Frizz(Operator):
    def __init__(
        self,
        frequency: float,
        amplitude: float,
        rand_frequency: Optional[float] = None,
        rand_amplitude: Optional[float] = None,
        rand_noise: float = 10.0,
        optimizable: bool = True,
        rand_mode: RandMode = RandMode.FIX,
        n_strands: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(rand_mode, generator)
        self.frequency = BoundedParameter(
            frequency,
            min_value=1e-2,
            max_value=1.0,
            generator=generator,
            requires_grad=optimizable,
        )
        self.amplitude = BoundedParameter(
            amplitude,
            min_value=0.0,
            generator=generator,
            requires_grad=optimizable,
        )
        self.rand_frequency = BoundedParameter(
            rand_frequency,
            default_value=0.0,
            min_value=0.0,
            generator=generator,
            requires_grad=optimizable,
        )
        self.rand_amplitude = BoundedParameter(
            rand_amplitude,
            default_value=0.0,
            min_value=0.0,
            generator=generator,
            requires_grad=optimizable,
        )
        self.rand_noise = rand_noise

        self.rand1 = RandomParameter(
            (n_strands, 1, 1),
            distribution="normal",
            requires_grad=rand_mode == RandMode.OPTIMIZE
            and optimizable
            and rand_frequency is not None,
        )
        self.rand2 = RandomParameter(
            (n_strands, 1, 1),
            distribution="uniform",
            requires_grad=False,  # Not differentiable for now
        )
        self.rand3 = RandomParameter(
            (n_strands, 1, 1),
            distribution="normal",
            requires_grad=rand_mode == RandMode.OPTIMIZE
            and optimizable
            and rand_amplitude is not None,
        )

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def forward(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = strands.device

        freq = self.frequency + self.rand_frequency * self.rand1

        length = compute_strand_lengths(strands).reshape(-1, 1, 1)
        u = (
            torch.arange(strands.shape[1], device=device) / (strands.shape[1] - 1.0)
        ).reshape(1, -1, 1)

        # Noise between [-0.5, 0.5]
        noise = 0.5 * fractal_noise3(freq * length * u + self.rand_noise * self.rand2)

        amp = self.amplitude + self.rand_amplitude * self.rand3

        offset = amp * noise
        offset[:, 0] = 0

        return strands + offset

    def mean(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return strands


class Bend(Operator):
    def __init__(
        self,
        angle: float,
        bend_start: float,
        rand_angle: Optional[float] = None,
        rand_root_angle: float = 1.0,
        bend_start_optimizable: bool = True,
        optimizable: bool = True,
        rand_mode: RandMode = RandMode.FIX,
        n_strands: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(rand_mode, generator)
        self.angle = BoundedParameter(
            angle,
            requires_grad=optimizable,
        )
        self.bend_start = BoundedParameter(
            bend_start,
            min_value=0.0,
            max_value=0.85,
            requires_grad=optimizable and bend_start_optimizable,
        )
        self.rand_angle = BoundedParameter(
            rand_angle,
            min_value=0.0,
            default_value=0.0,
            requires_grad=optimizable,
        )
        self.rand_root_angle = BoundedParameter(
            rand_root_angle,
            min_value=0.0,
            default_value=0.0,
            requires_grad=False,
        )

        self.rand = RandomParameter(
            (n_strands, 1, 1),
            distribution="normal",
            requires_grad=rand_mode == RandMode.OPTIMIZE
            and optimizable
            and rand_angle is not None,
        )

        self.rand2 = RandomParameter(
            (n_strands, 1, 1),
            distribution="normal",
            requires_grad=False,  # WTODO Make this work in opt
        )

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def forward(
        self,
        strands: torch.Tensor,
        guides: torch.Tensor,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = strands.device

        strand_angle = self.angle + self.rand_angle * self.rand

        if guides.dim() == 2:
            root_dir = guides.unsqueeze(1)
        else:
            root_dir = (guides[:, 1] - guides[:, 0]).unsqueeze(1)

        root_dir = torch.nn.functional.normalize(root_dir, dim=-1)

        seg_dir = strands[:, 1:] - strands[:, :-1]

        aux_axis = torch.where(
            torch.abs(root_dir[..., 1:2]) < 1e-1,
            torch.tensor([[[0.0, 1.0, 0.0]]], device=device),
            torch.tensor([[[1.0, 0.0, 0.0]]], device=device),
        )
        axis = torch.cross(root_dir, aux_axis, dim=-1)

        root_angle = self.rand_root_angle * self.rand2 * torch.pi
        root_rotation = axis_angle_rotation(root_dir, root_angle)
        root_rotation = root_rotation.expand(-1, seg_dir.shape[1], -1, -1)
        axis = torch.einsum("ijkl,ijl->ijk", root_rotation, axis)

        # No rotation if the segment direction and the base direction are exactly the same
        axis = torch.where(
            axis.norm(dim=-1)[..., None] > 1e-4,
            torch.nn.functional.normalize(axis, dim=-1),
            torch.tensor(0, device=device),
        )

        # Increasing rotation from root to tip
        seg_length_cumsum = seg_dir.norm(dim=-1, keepdim=True).cumsum(dim=1)
        v = seg_length_cumsum / seg_length_cumsum[:, -1:]
        # Smooth step from 0 to 1 at bend_start so it is differentiable
        smooth_start_step = (
            0.5 * torch.nn.functional.tanh(10.0 * (v - self.bend_start + 0.05)) + 0.5
        )
        seg_length = seg_dir.norm(dim=-1, keepdim=True) * smooth_start_step
        seg_length_cumsum = seg_length.cumsum(dim=1)
        v = seg_length_cumsum / seg_length_cumsum[:, -1:]
        seg_angle = v * strand_angle

        # Construct axis-angle rotation matrix
        rotation = axis_angle_rotation(axis, seg_angle)

        rotated_seg_dir = torch.einsum(
            "ijkl,ijlm->ijkm", rotation, seg_dir[..., None]
        ).squeeze(dim=-1)

        # Bend in lower segment affects upper segment, so accumulate directions
        result = strands.clone()
        result[:, 1:] = result[:, 0, None].broadcast_to(
            rotated_seg_dir.shape
        ) + rotated_seg_dir.cumsum(dim=1)

        return result

    def mean(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return strands


class Clump(Operator):
    def __init__(
        self,
        profile_a: float,
        curl_clump: bool = False,
        optimizable: bool = True,
    ):
        super().__init__()
        self.profile_a = BoundedParameter(
            profile_a,
            min_value=0,
            requires_grad=optimizable,
        )
        self.curl_clump = curl_clump

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def forward(
        self,
        strands: torch.Tensor,
        guides: torch.Tensor,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = strands.device

        u = (
            torch.arange(strands.shape[1], device=device) / (strands.shape[1] - 1.0)
        ).reshape(1, -1, 1)

        alpha = 1.0 - torch.exp(-self.profile_a * u)
        clumped = torch.lerp(strands, guides, alpha.broadcast_to(strands.shape))

        return clumped

    def mean(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # A curl clump is better approximated by the original strands than the clumped strands
        if self.curl_clump:
            return strands
        return self.forward(strands, guides, guide_idx)


# Based off Blender's Curl Hair Curves geometry node
class Curl(Operator):
    def __init__(
        self,
        curl_factor: float,
        radius: float,
        frequency: float,
        rand_frequency: Optional[float] = None,
        radius_factor_start: float = 0.430,
        radius_factor_end: float = 1.220,
        curl_start: float = 0.2,
        rand_start: Optional[float] = None,
        optimizable: bool = True,
        rand_mode: RandMode = RandMode.FIX,
        n_strands: Optional[int] = None,
        n_guides: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(rand_mode, generator)
        self.curl_factor = BoundedParameter(
            curl_factor,
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )
        self.radius = BoundedParameter(
            radius,
            min_value=1e-3,
            requires_grad=optimizable,
        )
        self.frequency = BoundedParameter(
            frequency,
            min_value=1e-3,
            requires_grad=optimizable,
        )
        self.rand_frequency = BoundedParameter(
            rand_frequency,
            default_value=0.0,
            min_value=0.0,
            requires_grad=optimizable,
        )
        self.rand_start = BoundedParameter(
            rand_start,
            min_value=0.0,
            max_value=1.0,
            default_value=0.0,
            requires_grad=False,
        )
        self.radius_factor_start = BoundedParameter(
            radius_factor_start, min_value=0.0, requires_grad=False
        )
        self.radius_factor_end = BoundedParameter(
            radius_factor_end, min_value=0.0, requires_grad=False
        )
        self.curl_start = BoundedParameter(curl_start, requires_grad=optimizable)

        self.rand1 = RandomParameter(
            (n_guides, 1, 1),
            distribution="normal",
            requires_grad=rand_mode == RandMode.OPTIMIZE
            and optimizable
            and rand_frequency is not None,
        )
        self.rand2 = RandomParameter(
            (n_guides, 1, 1),
            distribution="uniform",
            requires_grad=rand_mode == RandMode.OPTIMIZE and optimizable,
        )
        self.rand3 = RandomParameter(
            (n_guides, 1, 1),
            distribution="normal",
            requires_grad=rand_mode == RandMode.OPTIMIZE
            and optimizable
            and rand_start is not None,
        )

    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def forward(
        self,
        strands: torch.Tensor,
        guides: torch.Tensor,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = strands.device

        seg_dir = strands[:, 1:] - strands[:, :-1]
        seg_length = seg_dir.norm(dim=-1, keepdim=True)

        # Compute curl in local frame
        freq = 3.0 * (self.frequency + self.rand_frequency * self.rand1[guide_idx])
        freq = torch.clamp(freq, min=0)
        phase = torch.cumsum(seg_length * freq, dim=1) + 3.90 * self.rand2[guide_idx]
        length = compute_strand_lengths(strands).reshape(-1, 1, 1)
        v = seg_dir.norm(dim=-1, keepdim=True).cumsum(dim=1) / length

        radius = self.radius * (
            self.radius_factor_start
            + (self.radius_factor_end - self.radius_factor_start) * v
        )
        angle = 2 * torch.pi * phase
        curl_shape_x = radius * torch.sin(angle)
        curl_shape_y = radius * torch.cos(angle)

        # Curl strands around curl guide
        guide_dir = guides[:, 1:] - guides[:, :-1]
        guide_dir_padded = torch.cat([guide_dir, guide_dir[:, -1:]], dim=1)
        vert_tangent = torch.nn.functional.normalize(
            0.5 * (guide_dir_padded[:, 1:] + guide_dir_padded[:, :-1]), eps=1e-5, dim=-1
        )
        # "Z-up normal" https://github.com/blender/blender/blob/768c68f19b82c311fd67544f45e3edfd514b4075/source/blender/blenkernel/intern/curve_poly.cc#L121
        vert_normal = torch.where(
            torch.abs(vert_tangent[..., 0:1]) + torch.abs(vert_tangent[..., 1:2])
            < 1e-4,
            torch.tensor([[[1.0, 0.0, 0.0]]], device=device),
            torch.nn.functional.normalize(
                torch.stack(
                    [
                        vert_tangent[..., 1],
                        -vert_tangent[..., 0],
                        torch.zeros_like(vert_tangent[..., 2]),
                    ],
                    dim=-1,
                ),
                dim=-1,
            ),
        )
        vert_binormal = torch.cross(vert_tangent, vert_normal, dim=-1)

        # Smooth step from 0 to 1 at bend_start so it is differentiable
        smooth_start_step = (
            0.5
            * torch.nn.functional.tanh(
                5.0
                * (
                    v
                    - (self.curl_start + self.rand_start * self.rand3[guide_idx])
                    + 0.05
                )
            )
            + 0.5
        )
        curl_factor = self.curl_factor * smooth_start_step
        curl_offset = curl_factor * (
            curl_shape_x * vert_normal + curl_shape_y * vert_binormal
        )

        curled = strands.clone()
        curled[:, 1:] += curl_offset
        return curled

    def mean(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return strands
