# must install osmesa to run this script
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import trimesh
import pyrender
import imageio
import math
from tqdm import tqdm

# change these directories if you want
in_dir = "chairs/models/"
out_dir = "chairs/images/"


def normalize(buffer):  # this is currently not being used
    maximum = 0
    minimum = math.inf

    for row in buffer:
        for n in row:
            if maximum < n:
                maximum = n
            if n < minimum and 0 < n:
                minimum = n

    buffer = ((buffer - minimum) / (maximum - minimum))
    buffer[buffer < 0.0] = 1
    buffer = 1 - buffer

    return buffer


def fitted_normalize(buffer):
    buffer = (buffer - 0.05) * 1000
    buffer[buffer < 0] = 1
    buffer[1 < buffer] = 1
    buffer = 1 - buffer

    return buffer


def get_image_array(model_path, camera_pose):
    mesh = pyrender.Mesh.from_trimesh(trimesh.load(model_path))

    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)

    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(512, 512)
    color, depth = renderer.render(scene)

    depth_uint8 = (fitted_normalize(depth) * 255).astype(np.uint8)

    return depth_uint8


# a rotation and a translation one unit back
x_pose = np.array([
    [0, 0, -1, -1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])

y_pose = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])

z_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
])

if in_dir[-1] != "/":
    in_dir = in_dir + "/"

if out_dir[-1] != "/":
    out_dir = out_dir + "/"

for filename in tqdm(os.listdir(in_dir)):
    imageio.imwrite(
        out_dir + filename[0:filename.index(".")] + "_x.png",
        get_image_array(in_dir + filename, x_pose)
    )

    imageio.imwrite(
        out_dir + filename[0:filename.index(".")] + "_y.png",
        get_image_array(in_dir + filename, y_pose)
    )

    imageio.imwrite(
        out_dir + filename[0:filename.index(".")] + "_z.png",
        get_image_array(in_dir + filename, z_pose)
    )
