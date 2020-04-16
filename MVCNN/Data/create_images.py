from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import math
import imageio
import pyrender
import trimesh
import numpy as np
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


num_cores = multiprocessing.cpu_count()

# list of camera angles in pairs (x, y) where x and y are the angles to rotate around the x and y axis
camera_angles = [
    (-15, 30),  # 0
    (-15, 60),  # 1
    (-15, 0),  # 2
    (-15, -30),  # 3
    (-15, -60),  # 4
    (-15, 180)  # 5
    # z view
]

# change these directories if you want
in_dir = "./test_models/"
out_dir = "./images/"

# fix paths
if in_dir[-1] != "/":
    in_dir = in_dir + "/"

if out_dir[-1] != "/":
    out_dir = out_dir + "/"


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

    renderer = pyrender.OffscreenRenderer(64, 64)
    color, depth = renderer.render(scene)

    depth_uint8 = (fitted_normalize(depth) * 255).astype(np.uint8)

    return depth_uint8


def get_camera_transformation(x_angle, y_angle):
    x_angle = math.radians(x_angle)
    y_angle = math.radians(y_angle)

    # translates 1 unit back
    translation = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])

    # x_angle is the angle to rotate around the x axis
    x_axis_rotation = np.array([
        [1, 0, 0, 0],
        [0, math.cos(x_angle), -math.sin(x_angle), 0],
        [0, math.sin(x_angle), math.cos(x_angle), 0],
        [0, 0, 0, 1]
    ])

    # y_angle is the angle to rotate around the y axis
    y_axis_rotation = np.array([
        [math.cos(y_angle), 0, math.sin(y_angle), 0],
        [0, 1, 0, 0],
        [-math.sin(y_angle), 0, math.cos(y_angle), 0],
        [0, 0, 0, 1]
    ])

    return y_axis_rotation.dot(x_axis_rotation).dot(translation)


def parallel_function(filename):
    for i in range(len(camera_angles)):
        x_angle, y_angle = camera_angles[i]

        in_path = in_dir + filename
        out_path = out_dir + \
            filename[0:filename.index(".")] + "_" + str(i) + ".png"

        imageio.imwrite(
            out_path,
            get_image_array(
                in_path, get_camera_transformation(x_angle, y_angle))
        )


l = tqdm(os.listdir(in_dir))
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(parallel_function)(i) for i in l)
