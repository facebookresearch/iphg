# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import itertools
from typing import List, Mapping, Optional, Any, Dict
from .operator import Operator, BoundedParameter
from .guides import Instance


class OperatorChain(torch.nn.Module):
    def __init__(
        self,
        operators: List[Operator],
        optimizable: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()

        # Check there is exactly one instance operator
        assert sum([isinstance(op, Instance) for op in operators]) == 1

        self.operators = operators
        self.generator = generator

        if generator is not None:
            for operator in self.operators:
                operator.generator = self.generator

        if not optimizable:
            for param in self.parameters():
                param.requires_grad_(False)

    def state_dict(self):
        state_dict = super().state_dict()
        return {
            **state_dict,
            "operators": [operator.state_dict() for operator in self.operators],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        state_dict = state_dict.copy()
        operators = state_dict.pop("operators")
        for operator, operator_state_dict in zip(self.operators, operators):
            operator.load_state_dict(operator_state_dict, strict)
        return super().load_state_dict(state_dict, strict)

    def to(self, device):
        super().to(device)
        for operator in self.operators:
            operator.to(device)
        return self

    def forward(
        self,
        roots: torch.Tensor,
        guides: torch.Tensor,
        guide_normals: torch.Tensor,
        mean: bool = False,
    ) -> torch.Tensor:
        # Find the index of the instance operator
        inst_idx = [isinstance(op, Instance) for op in self.operators].index(True)
        for operator in self.operators[:inst_idx]:
            guides = operator(guides, guide_normals, mean=mean)

        instance: Instance = self.operators[inst_idx]
        instance.update(roots, guides)
        blended_guides = instance.compute_blended_guides(roots, guides)
        operator_guides, operator_guide_idx = instance.compute_operator_guides(
            roots, guides
        )
        strands = instance(roots, blended_guides)

        for operator in self.operators[inst_idx + 1 :]:
            strands = operator(
                strands,
                operator_guides,
                operator_guide_idx,
                mean=mean,
            )

        return strands

    def get_instance(self) -> Instance:
        return next(filter(lambda op: isinstance(op, Instance), self.operators))

    def parameters(self) -> List[BoundedParameter]:
        return list(
            itertools.chain.from_iterable(
                [operator.parameters() for operator in self.operators]
            )
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> List[BoundedParameter]:
        return list(
            itertools.chain.from_iterable(
                [
                    operator.named_parameters(prefix, recurse)
                    for operator in self.operators
                ]
            )
        )

    def get_optimizable_parameters(
        self, filter_dict: Dict[str, bool] = {}
    ) -> List[BoundedParameter]:
        random = filter_dict.pop("random", None)

        assert all(filter_dict.values()) or not any(
            filter_dict.values()
        ), "filter_dict cannot both include (True) and exclude (False) operators."

        include_mode = all(filter_dict.values()) and len(filter_dict) > 0
        exclude_mode = not include_mode

        params = []
        for operator in self.operators:
            if include_mode and operator.__class__.__name__ not in filter_dict:
                continue

            if exclude_mode and operator.__class__.__name__ in filter_dict:
                continue

            for param in operator.get_optimizable_parameters():
                if random == True and not param.is_random():
                    continue
                if random == False and param.is_random():
                    continue

                params.append(param)

        return params
