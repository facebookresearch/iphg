# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bpy
import numpy as np
import os
import sys
import json
from mathutils import Matrix
from bpy_extras.io_utils import axis_conversion

argv = sys.argv
argv = argv[argv.index("--args") + 1 :]

dataset_path = argv[0]
dataset = argv[1]
in_dir = argv[2]
out_dir = argv[3]
view = int(argv[4])

out_dir = os.path.join(out_dir, "images")
os.makedirs(out_dir, exist_ok=True)

is_gh = "gh" in dataset


def enable_gpus():
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = list(cycles_preferences.devices)[:2]

    activated_gpus = []

    for device in devices:
        device.use = True
        activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = "OPTIX"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True

    return activated_gpus


enable_gpus()


def load_strands(verts, name, vis=False, thickness=0.15):
    # Create the curve and set its points
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    for strand in verts:
        polyline = curve_data.splines.new("POLY")
        polyline.points.add(len(strand) - 1)
        polyline.points.foreach_set(
            "co", np.concatenate((strand, np.ones((len(strand), 1))), axis=1).flatten()
        )

    curve_object = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_object)

    curve_object.select_set(True)
    bpy.context.view_layer.objects.active = curve_object
    bpy.ops.object.convert(target="CURVES")

    mat = bpy.data.materials.get("Head")
    principled = mat.node_tree.nodes["Principled BSDF"]

    if vis:
        mat = bpy.data.materials.get("Visualization")
        bpy.context.scene.world = bpy.data.worlds["constant"]
        principled.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1)
    else:
        mat = bpy.data.materials.get("Blond hair")
        hair_bsdf = mat.node_tree.nodes["Principled Hair BSDF"]
        hair_bsdf.inputs["Melanin"].default_value = 0.4
        hair_bsdf.inputs["Melanin Redness"].default_value = 0.7
        hair_bsdf.inputs["Random Color"].default_value = 0.3
        hair_bsdf.inputs["IOR"].default_value = 2.0
        bpy.context.scene.world = bpy.data.worlds["constant"]
        bpy.data.worlds["constant"].node_tree.nodes["Emission"].inputs[
            "Strength"
        ].default_value = 1.0
        principled.inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1)
    curve_object.data.materials.append(mat)

    radius = curve_object.data.attributes["radius"].data
    for i in range(len(radius)):
        if "guides" in name:
            val = thickness * 3e-3
        else:
            val = thickness * 1e-3

        radius[i].value = val


if not is_gh:
    file_path = os.path.join(dataset_path, "cam_params.json")
    with open(file_path) as json_file:
        camera_dict = json.loads(json_file.read())

    cam_data_list = camera_dict["cam_list"]
    cam_data = cam_data_list[view]

    pose = np.array(cam_data["pose"])
    fovx = 2 * np.arctan2(1, cam_data["ndc_prj"][0])

    width = 1000
    height = int(width * cam_data["ndc_prj"][0] / cam_data["ndc_prj"][1])
else:
    cam_data = np.load(os.path.join(in_dir, "camera.npz"))
    K, pose = cam_data["K"], cam_data["pose"]

    pose[:3, 1:3] *= -1

    def focal2fov(focal, pixels):
        return 2 * np.arctan2(pixels, (2 * focal))

    fx = K[0, 0]
    fy = K[1, 1]
    fovx = 2 * focal2fov(fx, 1)

    width = 1000
    height = int(width * fx / fy)

global_matrix = np.array(axis_conversion(to_forward="-Z", to_up="Y").to_4x4())

scn = bpy.context.scene

cam = bpy.data.cameras.new(f"Camera real")
cam.sensor_fit = "HORIZONTAL"
cam.angle_x = fovx
cam.shift_x = 0
cam.shift_y = 0
cam_obj = bpy.data.objects.new(f"Camera real", cam)

matrix = np.linalg.inv(global_matrix) @ pose
cam_obj.matrix_world = Matrix(matrix)

scn.collection.objects.link(cam_obj)

bpy.context.scene.camera = cam_obj
bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height
bpy.context.scene.render.image_settings.color_mode = "RGB"


objs = bpy.data.objects
objs.remove(objs["head"], do_unlink=True)
objs.remove(objs["eyes"], do_unlink=True)
bpy.ops.wm.obj_import(filepath=os.path.join(dataset_path, "head_flame.obj"))

if not is_gh:
    model_tsfm = (
        np.fromfile(os.path.join(dataset_path, "model_tsfm.dat"), dtype=np.float32)
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
    transform = np.linalg.inv(global_matrix) @ model_tsfm @ translation
    objs["head_flame"].matrix_world = Matrix(transform)

objs["head_flame"].data.materials.append(bpy.data.materials.get("Head"))

bpy.ops.object.shade_smooth()
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.device = "GPU"

data = np.load(os.path.join(in_dir, "final.npz"))

for name, verts in data.items():
    load_strands(
        verts,
        name,
        vis="guides" in name,
        thickness=0.6 if is_gh else 0.15,
    )
    objs[name].matrix_world = Matrix(np.linalg.inv(global_matrix))

    bpy.context.scene.render.filepath = os.path.join(out_dir, f"{name}.png")
    bpy.ops.render.render(write_still=True)

    objs = bpy.data.objects
    objs.remove(objs[name], do_unlink=True)
