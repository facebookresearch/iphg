# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, Literal
from enum import IntEnum
from .noise import trunc_normal
import torch


class RandMode(IntEnum):
    SAMPLE = 0
    FIX = 1
    OPTIMIZE = 2


class BoundedParameter(torch.nn.Parameter):
    def __new__(
        cls,
        param: Union[float, torch.Tensor],
        default_value: Optional[float] = None,
        requires_grad: bool = True,
        *args,
        **kwargs,
    ):
        if param is None and default_value is None:
            raise ValueError("Default value must be set if param is None!")

        if param is None:
            value = torch.tensor(default_value, dtype=torch.float32)
        elif isinstance(param, torch.Tensor):
            value = param
        else:
            value = torch.tensor(param, dtype=torch.float32)

        return super().__new__(
            cls, data=value, requires_grad=requires_grad and param is not None
        )

    def __init__(
        self,
        param: Union[float, torch.Tensor],
        default_value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        requires_grad: bool = True,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.generator = generator

    def clamp_bounds(self):
        with torch.no_grad():
            if self.min_value is not None:
                self.data.clamp_min_(self.min_value)
            if self.max_value is not None:
                self.data.clamp_max_(self.max_value)

    def is_random(self):
        return False


class RandomParameter(BoundedParameter):
    def __new__(
        cls,
        shape: torch.Size,
        requires_grad: bool = True,
        *args,
        **kwargs,
    ):
        return super().__new__(
            cls, torch.zeros(shape), requires_grad=requires_grad, *args, **kwargs
        )

    def __init__(
        self,
        shape: torch.Size,
        distribution: Union[Literal["uniform"], Literal["normal"]] = "uniform",
        generator: Optional[torch.Generator] = None,
        requires_grad: bool = True,
    ):
        assert distribution in ["uniform", "normal"]

        super().__init__(
            None,
            None,
            0 if distribution == "uniform" else -1,
            1.0,
            generator,
            requires_grad,
        )
        self.initialized = False
        self.distribution = distribution

    def is_random(self):
        return self.requires_grad

    @torch.no_grad()
    def resample(self, generator: Optional[torch.Generator] = None, device=None):
        if self.distribution == "uniform":
            self[...] = torch.rand(
                self.shape,
                generator=generator,
                device=device,
            )
        elif self.distribution == "normal":
            trunc_normal(self, 0, 1, -1, 1, generator)
        else:
            raise ValueError("Invalid distribution!")

        self.initialized = True


class Operator(torch.nn.Module):
    def __init__(
        self,
        rand_mode: RandMode = RandMode.OPTIMIZE,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.generator = generator
        self.rand_mode = rand_mode

    def state_dict(self):
        state_dict = super().state_dict()
        return {**state_dict, "rand_mode": self.rand_mode}

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()
        self.rand_mode = state_dict.pop("rand_mode")
        for param in self.parameters():
            if isinstance(param, RandomParameter):
                param.initialized = True  # WTODO Don't have a good way of setting this
        return super().load_state_dict(state_dict, strict)

    def mean(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError("Mean computation not implemented for this operator.")

    def __call__(
        self,
        strands: torch.Tensor,
        guides: Optional[torch.Tensor] = None,
        guide_idx: Optional[torch.Tensor] = None,
        mean: bool = False,
    ) -> torch.Tensor:
        self.update_random_parameters()
        if mean:
            return self.mean(strands, guides, guide_idx)
        return self.forward(strands, guides, guide_idx)

    def get_optimizable_parameters(self) -> List[BoundedParameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def clamp_parameters(self):
        for param in self.parameters():
            param.clamp_bounds()

    def update_random_parameters(self):
        for param in self.parameters():
            if isinstance(param, RandomParameter):
                if not param.initialized or self.rand_mode == RandMode.SAMPLE:
                    param.resample(self.generator, param.device)
