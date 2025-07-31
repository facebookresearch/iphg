# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import nvdiffrast.torch as dr
import gc
import math
import matplotlib.pyplot as plt
from typing import Optional, Dict
from dataset_readers import Camera
from head import HeadModel


def repeat_structure(struct, n_inc_size, n_repeat):
    return (
        struct.repeat(n_repeat, 1)
        + torch.arange(n_repeat, device="cuda", dtype=struct.dtype)
        .repeat_interleave(len(struct))
        .unsqueeze(1)
        * n_inc_size
    )


class Renderer:
    def __init__(
        self,
        head: Optional[HeadModel] = None,
        hair_width: float = 7e-4,
    ) -> None:
        self.glctx = dr.RasterizeCudaContext()

        if head is not None:
            head_extent_length = head.extent_length
        else:
            head_extent_length = 1.0

        self.hair_width = hair_width * head_extent_length
        self.head = head

        self.debug_colors = (
            torch.tensor(
                [
                    [210, 210, 230],
                    [200, 200, 210],
                    [153, 102, 204],
                    [204, 204, 255],
                    [0, 200, 0],
                    [0, 180, 50],
                    [200, 110, 0],
                ]
            ).cuda()
            / 255
        )

    # Generate triangle indices for hair strips and head
    def _generate_tri(self, strands):
        n, k, _ = strands.shape
        quad = torch.tensor(
            [[0, 1, 1 + n * k], [0, 1 + n * k, n * k]], dtype=torch.int32, device="cuda"
        )
        strip = repeat_structure(quad, 1, k - 1)
        tri = repeat_structure(strip, k, n)
        if self.head is not None:
            tri = torch.cat([self.head.tri + 2 * n * k, tri], axis=0)

        return tri.contiguous()

    # Generate hair vertices of connected quads that always face the camera
    @torch.compile(disable=torch.cuda.get_device_properties(0).major < 7)
    def _generate_hair_vert(self, cam_center, strands):
        seg_dir = torch.nn.functional.normalize(
            strands[:, 1:] - strands[:, :-1], dim=-1
        )
        seg_center = 0.5 * (strands[:, 1:] + strands[:, :-1])
        view_dir = torch.nn.functional.normalize(
            cam_center[None, None, :] - seg_center, dim=-1
        )

        seg_binormal = torch.cross(seg_dir, view_dir, dim=-1)
        aux_axis = torch.where(
            torch.abs(view_dir[:, :, 1:2]) < 1e-1,
            torch.tensor([[[0.0, 1.0, 0.0]]], device="cuda"),
            torch.tensor([[[1.0, 0.0, 0.0]]], device="cuda"),
        )
        # Ensure binormal is not of length 0, which occurs when view_dir is parallel to seg_dir
        seg_binormal_norm = torch.norm(seg_binormal, dim=-1, keepdim=True)
        seg_binormal = torch.where(
            seg_binormal_norm > 1e-5,
            seg_binormal / (seg_binormal_norm + 1e-8),
            torch.nn.functional.normalize(
                torch.cross(aux_axis, view_dir, dim=-1), dim=-1
            ),
        )

        seg_binormal_padded = torch.cat(
            [seg_binormal[:, 0:1], seg_binormal, seg_binormal[:, -1:]], axis=1
        )
        vert_binormal = 0.5 * (seg_binormal_padded[:, :-1] + seg_binormal_padded[:, 1:])
        vert_binormal_norm = torch.norm(vert_binormal, dim=-1, keepdim=True)
        # If binormal is still length 0, this means adjacent segments are pointing
        # in opposite directions, so simply use the binormal of the previous segment
        vert_binormal = torch.where(
            vert_binormal_norm > 1e-5,
            vert_binormal / (vert_binormal_norm + 1e-8),
            seg_binormal_padded[:, :-1],
        )

        # Generate connected quad vertices
        return torch.cat(
            [
                strands - vert_binormal * self.hair_width,
                strands + vert_binormal * self.hair_width,
            ],
            axis=0,
        ).reshape(-1, 3)

    def render(
        self,
        cam: Camera,
        strands: torch.Tensor,
        hair_color_idx: Optional[torch.Tensor] = None,
        render_head: bool = True,
        n_strands_batch_size: int = 80000,
    ) -> Dict[str, torch.Tensor]:
        def transform_pos(mtx, pos):
            # (x,y,z) -> (x,y,z,1)
            posw = torch.cat(
                [pos, torch.ones([pos.shape[0], 1], device="cuda")], axis=1
            )
            return (posw @ mtx).unsqueeze(0)

        # Cuda rasterizer can't handle too many strands at once, so we split them into chunks
        n_chunks = math.ceil(len(strands) / n_strands_batch_size)
        strand_chunks = torch.chunk(strands, n_chunks)

        final_img = None
        final_depth = None

        for i, strands in enumerate(strand_chunks):
            if self.head is not None:
                strands = self.head.scale_from_head(strands)
            vert = self._generate_hair_vert(cam.camera_center, strands)
            if self.head is not None and render_head:
                head_vert = self.head.scale_from_head(self.head.vert)
                vert = torch.cat([vert, head_vert], axis=0)
            vert = vert.contiguous()

            tri = self._generate_tri(strands)

            if hair_color_idx is None:
                hair_color = torch.zeros(len(strands), 3, device="cuda")
                for i, color in enumerate(self.debug_colors):
                    hair_color[i :: len(self.debug_colors)] = color
            else:
                hair_color_idx_chunk = hair_color_idx[
                    i * n_strands_batch_size : (i + 1) * n_strands_batch_size
                ]
                hair_color = self.debug_colors[
                    hair_color_idx_chunk % len(self.debug_colors)
                ]

            hair_color = hair_color.repeat_interleave(strands.shape[1], dim=0)
            strand_vertex_color = torch.reshape(hair_color, (-1, 3))
            # Duplicate color since quad vertices have 2 vertices per strand vertex
            color = torch.cat([strand_vertex_color, strand_vertex_color], axis=0)

            if self.head is not None and render_head:
                head_color = self.head.compute_head_color(cam.camera_center)
                color = torch.cat([color, head_color], axis=0)

            color = color.clamp(0.0, 1.0).contiguous()

            pos_clip = transform_pos(cam.full_proj_transform, vert).contiguous()

            rast, _ = dr.rasterize(
                self.glctx,
                pos_clip,
                tri,
                resolution=[cam.image_height, cam.image_width],
            )
            img, _ = dr.interpolate(color, rast, tri)
            img = dr.antialias(img, rast, pos_clip, tri)

            if len(strand_chunks) > 1:
                depth = rast[..., 2]
                triangle_id = rast[..., 3]
                depth = depth.clone()
                depth[triangle_id == 0] = 1e10  # Set background depth to a large value

                if final_depth is None:
                    final_depth = depth
                    final_img = img
                else:
                    final_img = torch.where(
                        (depth <= final_depth).unsqueeze(-1),
                        img,
                        final_img,
                    )
                    final_depth = torch.minimum(final_depth, depth)
            else:
                final_img = img

        return final_img.squeeze(0).permute(2, 0, 1)


def render_grooms(renderer, camera, *grooms, save=False, save_path=None):
    grooms = [
        g
        for g in grooms
        if g is not None
        and (torch.is_tensor(g) or (type(g) == dict and g["strands"] is not None))
    ]
    aspect = camera.image_height / camera.image_width
    fig, ax = plt.subplots(
        1, len(grooms), figsize=(10 * len(grooms), 10 * aspect), dpi=100
    )
    if len(grooms) == 1:
        ax = [ax]
    with torch.no_grad():
        for i, groom in enumerate(grooms):
            if torch.is_tensor(groom):
                strands = groom
                cluster_idx = None
                title = None
            elif type(groom) == dict:
                strands = groom["strands"]
                cluster_idx = groom.get("cluster_idx", None)
                title = groom.get("title", None)
            image = renderer.render(
                camera,
                strands,
                hair_color_idx=cluster_idx,
            )
            image = torch.clamp(image, min=0, max=1)
            ax[i].imshow(image.permute(1, 2, 0).cpu().numpy())
            ax[i].axis("off")
            ax[i].set_title(title)
            ax[i].title.set_size(40)
    plt.tight_layout()
    plt.show()
    if save_path is not None and save:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    # https://github.com/matplotlib/matplotlib/issues/27713
    gc.collect()
