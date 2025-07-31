# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Callable, List, Any
from head import HeadModel
from grooming_graph.operators import OperatorChain
from grooming_graph.strands import Strands
from renderer import Renderer
from dataset_readers import Camera


class Scene:
    def __init__(
        self,
        head: HeadModel,
        renderer: Renderer,
        render_fn: Callable[[Any, str], None],
        train_cameras: List[Camera],
        is_synthetic: bool,
        grooming_chain: OperatorChain,
        target_groom: torch.Tensor,
        target_guides: torch.Tensor,
        opt_strand_idx: torch.Tensor,
        roots: torch.Tensor,
        cluster_guides: torch.Tensor,
        initial_guides: torch.Tensor,
        guide_roots: torch.Tensor,
        guide_strands: Strands,
    ):
        self.head = head
        self.renderer = renderer
        self.train_cameras = train_cameras
        self.is_synthetic = is_synthetic
        self.render_fn = render_fn
        self.grooming_chain = grooming_chain
        self.target_groom = target_groom
        self.target_guides = target_guides
        self.opt_strand_idx = opt_strand_idx
        self.roots = roots
        self.cluster_guides = cluster_guides
        self.initial_guides = initial_guides
        self.guide_roots = guide_roots
        self.guide_strands = guide_strands
