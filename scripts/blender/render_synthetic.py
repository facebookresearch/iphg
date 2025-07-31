# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bpy
import numpy as np
import os
import sys

argv = sys.argv
argv = argv[argv.index("--args") + 1 :]

dataset = argv[1]
in_dir = argv[2]
out_dir = argv[3]
frame_n = int(argv[4])
teaser = False
if len(argv) > 5:
    teaser = argv[5] == "True"

out_dir = os.path.join(out_dir, "images")
os.makedirs(out_dir, exist_ok=True)


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


def load_strands(verts, name, vis=False):
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
        if teaser:
            hair_bsdf.inputs["Melanin"].default_value = 6.0
            hair_bsdf.inputs["Melanin Redness"].default_value = 0.2
            hair_bsdf.inputs["Roughness"].default_value = 0.4
            hair_bsdf.inputs["Random Color"].default_value = 0.5
            hair_bsdf.inputs["IOR"].default_value = 1.7
            bpy.context.scene.world = bpy.data.worlds["env_map"]
            bpy.data.worlds["env_map"].node_tree.nodes["Background"].inputs[
                "Strength"
            ].default_value = 2.0
        else:
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
            val = 3e-3
        elif teaser and name == "edited_groom2":
            val = 2e-3
        else:
            val = 1e-3

        radius[i].value = val


bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.device = "GPU"
bpy.context.scene.frame_set(frame_n)
if not teaser:
    bpy.context.scene.render.image_settings.color_mode = "RGB"
else:
    bpy.context.scene.render.resolution_x = 1500
    bpy.context.scene.render.resolution_y = 1500

data = np.load(os.path.join(in_dir, "final.npz"))

for name, verts in data.items():
    verts[:, :, (0, 2, 1)] = verts[:, :, (0, 1, 2)]
    verts[:, :, 1] *= -1
    load_strands(verts, name, vis="guides" in name)

    bpy.context.scene.render.filepath = os.path.join(out_dir, f"{name}.png")
    bpy.ops.render.render(write_still=True)

    objs = bpy.data.objects
    objs.remove(objs[name], do_unlink=True)
