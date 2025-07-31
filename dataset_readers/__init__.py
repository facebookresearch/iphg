# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import json
import cv2
import math
import pickle
import torch
from plyfile import PlyData

from grooming_graph.utils.hair import resample_strands
from head import HeadModel


class Camera:
    def __init__(
        self,
        width,
        height,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(world_view_transform)
        self.camera_center = view_inv[3][:3]


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


# From https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py
def getProjectionMatrix(znear, zfar, fovX, fovY, x_offset=0, y_offset=0):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = x_offset
    P[1, 2] = y_offset
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def round_to_next_multiple(x, multiple):
    return int(math.ceil(x / multiple) * multiple)


def readSyntheticCameras(path: str, transformsfile: str):
    zfar = 100
    znear = 0.1

    file_path = os.path.join(path, transformsfile)
    with open(file_path) as json_file:
        camera_dict = json.loads(json_file.read())

    fovx = camera_dict["camera_angle_x"]
    frames = camera_dict["frames"]
    cameras = []

    for frame in frames:
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform
        w2c = torch.from_numpy(np.linalg.inv(c2w)).float().transpose(0, 1).cuda()

        width = 800
        height = 800

        if "camera_angle_x" in frame:
            fovx = frame["camera_angle_x"]
        fovy = focal2fov(fov2focal(fovx, width), height)

        proj = (
            getProjectionMatrix(
                znear=znear,
                zfar=zfar,
                fovX=fovx,
                fovY=fovy,
            )
            .transpose(0, 1)
            .cuda()
        )

        cam = Camera(
            width,
            height,
            w2c,
            w2c @ proj,
        )
        cameras.append(cam)

    return cameras


def readMonoHairCameras(path: str, transformsfile: str) -> None:
    zfar = 100
    znear = 0.1

    file_path = os.path.join(path, transformsfile)
    with open(file_path) as json_file:
        camera_dict = json.loads(json_file.read())

    cam_data_list = camera_dict["cam_list"]
    cameras = []

    for cam_data in cam_data_list:
        c2w = np.array(cam_data["pose"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        w2c = torch.from_numpy(np.linalg.inv(c2w)).float().transpose(0, 1).cuda()

        fovx = 2 * np.arctan2(1, cam_data["ndc_prj"][0])
        fovy = 2 * np.arctan2(1, cam_data["ndc_prj"][1])

        width = 1000
        height = int(width * cam_data["ndc_prj"][0] / cam_data["ndc_prj"][1])
        height = round_to_next_multiple(height, 8)

        proj = (
            getProjectionMatrix(
                znear=znear,
                zfar=zfar,
                fovX=fovx,
                fovY=fovy,
            )
            .transpose(0, 1)
            .cuda()
        )

        cam = Camera(
            width,
            height,
            w2c,
            w2c @ proj,
        )
        cameras.append(cam)

    return cameras


def readGaussianHaircutCameras(path: str, transformsfile: str) -> None:
    zfar = 100
    znear = 0.1

    file_path = os.path.join(path, transformsfile)
    camera_data = pickle.load(open(file_path, "rb"))

    cameras = []

    for cam_data in camera_data.values():
        intrinsics, pose = load_K_Rt_from_P(cam_data.transpose(0, 1)[:3, :4].numpy())

        w2c = torch.from_numpy(np.linalg.inv(pose)).float().transpose(0, 1).cuda()

        K = intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        fovx = 2 * focal2fov(fx, 1)
        fovy = 2 * focal2fov(fy, 1)

        width = 1000
        height = int(width * fx / fy)
        height = round_to_next_multiple(height, 8)

        proj = (
            getProjectionMatrix(
                znear=znear,
                zfar=zfar,
                fovX=fovx,
                fovY=fovy,
            )
            .transpose(0, 1)
            .cuda()
        )

        cam = Camera(
            width,
            height,
            w2c,
            w2c @ proj,
        )
        cameras.append(cam)

    return cameras


def load_synthetic_groom_guides(head: HeadModel, dirname: str, n_strand_segments=100):
    guides_path = os.path.join(dirname, "guides.npy")
    target_guides = torch.from_numpy(np.load(guides_path)).cuda()
    target_guides = head.scale_to_head(target_guides)
    target_guides = head.project_strands_onto_scalp(target_guides)
    target_guides = resample_strands(target_guides, n_strand_segments + 1)

    groom_path = os.path.join(dirname, "groom.npy")
    target_groom = torch.from_numpy(np.load(groom_path)).cuda()
    target_groom = head.scale_to_head(target_groom)
    target_groom = head.project_strands_onto_scalp(target_groom)
    target_groom = resample_strands(target_groom, n_strand_segments + 1)

    return target_guides, target_groom


def load_gaussian_haircut_groom(head: HeadModel, dirname: str):
    target_groom = torch.from_numpy(
        np.load(os.path.join(dirname, "strands.npy"))
    ).cuda()
    target_groom = target_groom[..., [0, 2, 1]]
    target_groom[..., 2] *= -1
    target_groom = head.scale_to_head(target_groom)
    target_groom = head.project_strands_onto_scalp(target_groom)

    return target_groom


def load_monohair_transform(dirname: str):
    model_tsfm = (
        np.fromfile(os.path.join(dirname, "model_tsfm.dat"), dtype=np.float32)
        .reshape(4, 4)
        .T
    )
    translation = np.array(
        [
            [1.0, 0.0, 0.0, 0.006],
            [0.0, 1.0, 0.0, -1.644],
            [0.0, 0.0, 1.0, 0.010],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transform = (model_tsfm @ translation).tolist()
    return transform, translation


def load_monohair_groom(head: HeadModel, dirname: str, translation, n_strand_segments):
    plydata = PlyData.read(os.path.join(dirname, "strands.ply"))
    vertices = plydata["vertex"]

    vert = torch.stack(
        [
            torch.tensor(vertices["x"]),
            torch.tensor(vertices["y"]),
            torch.tensor(vertices["z"]),
        ],
        dim=1,
    ).cuda()
    vert = torch.cat((vert, torch.ones_like(vert[:, 0:1])), dim=1)
    mtx = torch.from_numpy(translation).cuda()
    vert = (vert @ mtx.t())[:, :3].cuda()
    vert = head.scale_to_head(vert)

    outside = (
        head.compute_distance_to_head(vert).reshape(-1, n_strand_segments + 1) > -1e-4
    ).float()
    outside[outside == 0] = n_strand_segments + 1
    first_outside_idx = outside.argmin(dim=1)
    all_in_or_out = (first_outside_idx == 0) | (first_outside_idx >= n_strand_segments)
    target_groom = vert.reshape(-1, n_strand_segments + 1, 3)
    target_groom[~all_in_or_out] = resample_strands(
        target_groom[~all_in_or_out],
        n_strand_segments + 1,
        first_outside_idx[~all_in_or_out],
    )
    target_groom = head.project_strands_onto_scalp(target_groom)

    return target_groom
