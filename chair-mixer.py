import os
import json
import trimesh
import trimesh.util
import trimesh.transformations
import random
import numpy as np
from tqdm import tqdm
import math

import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

in_dir = "final-chairs/modified-input/a/"
out_dir = "final-chairs/output/objs/"


chair_dir_paths = list(map(
    lambda dir: in_dir + dir,
    os.listdir(in_dir)
))


def get_objs_from_json(chair_json):
    if "objs" in chair_json:
        return chair_json["objs"]

    if "children" in chair_json:
        objs = []

        for child in chair_json["children"]:
            objs += get_objs_from_json(child)

        return objs


def get_part_mesh(part_name, chair_path):
    chair_json = json.load(open(chair_path + "/result.json"))

    for chair_part_json in chair_json[0]["children"]:
        if chair_part_json["text"] == part_name:
            objs = get_objs_from_json(chair_part_json)
            mesh = trimesh.Trimesh()

            for obj in objs:
                obj_path = chair_path + "/objs/" + obj + ".obj"
                mesh = trimesh.util.concatenate(mesh, trimesh.load(obj_path))

            return mesh


def get_top_center(bounds):
    return ((bounds[0][0], bounds[1][1], bounds[0][2]) + bounds[1]) / 2


def get_bottom_center(bounds):
    return ((bounds[1][0], bounds[0][1], bounds[1][2]) + bounds[0]) / 2 


def reorient_part(mesh):
    inverse_transform = np.linalg.inv(mesh.bounding_box_oriented.primitive.transform)
    mesh.apply_transform(inverse_transform)


def fix_chair_translation(back_mesh, base_mesh, seat_mesh):
    # translate base
    random_translation = (0, (random.random() - 0.5) * 0.1, (random.random() - 0.5) * 0.2)
    seat_middle = ((seat_mesh.bounds[0] + seat_mesh.bounds[1]) / 2)
    translation = seat_middle - get_top_center(base_mesh.bounds)
    base_mesh.apply_translation(translation + random_translation)

    # translate back
    random_translation = (0, -random.random() * 0.1, (random.random()) * 0.1)
    seat_middle = ((seat_mesh.bounds[0] + seat_mesh.bounds[1]) / 2)
    y_translation = seat_middle[1] - get_bottom_center(back_mesh.bounds)[1]
    back_mesh.apply_translation((0, y_translation + random_translation[1], random_translation[2]))

    # # Lower back part
    # back_translation = (
    #     0, -back_mesh.bounds[0][1] + seat_mesh.bounds[1][1] - 0.04, 0
    # )
    # back_mesh.apply_translation(back_translation)


def get_x_length(bounds):
    return np.linalg.norm((bounds[0][0], bounds[1][1], bounds[1][2]) - bounds[1])


def get_z_length(bounds):
    return np.linalg.norm((bounds[1][0], bounds[1][1], bounds[0][2]) - bounds[1])


def fix_chair_scale(back_mesh, base_mesh, seat_mesh):
    # delta_xz_seat = seat_mesh.bounds[1] - seat_mesh.bounds[0]
    # delta_xz_base = base_mesh.bounds[1] - base_mesh.bounds[0]
    # scale = delta_xz_seat / delta_xz_base

    # scale back
    random_scale = (random.random() - 0.5) / 2  # between -0.25 and 0.25
    scale = get_x_length(seat_mesh.bounds) / get_x_length(back_mesh.bounds)
    scale_matrix = np.array([
        [scale + random_scale, 0, 0, 0],
        [0, scale + random_scale + (random.random()), 0, 0],
        [0, 0, scale + random_scale, 0],
        [0, 0, 0, 1]
    ])
    back_mesh.apply_transform(scale_matrix)

    # scale base
    random_scale = random.random() / 4  # between 0 and 0.25
    x_scale = get_x_length(seat_mesh.bounds) / get_x_length(base_mesh.bounds)
    z_scale = get_z_length(seat_mesh.bounds) / get_z_length(base_mesh.bounds)
    scale = min(x_scale, z_scale)
    scale_matrix = np.array([
        [scale - random_scale, 0, 0, 0],
        [0, scale - random_scale, 0, 0],
        [0, 0, scale - random_scale, 0],
        [0, 0, 0, 1]
    ])
    base_mesh.apply_transform(scale_matrix)



def break_chairs(back_mesh, base_mesh, seat_mesh):
    back_scale = (random.random() - 0.5) / 1.5 + 1
    back_translation = (0, (random.random() - 0.5) / 2,
                        (random.random() - 0.5) / 2)
    back_rotation = trimesh.transformations.rotation_matrix(
        math.radians((random.random() - 0.5) * 60),
        (1, 0, 0),
        trimesh.bounds.corners(back_mesh.bounds)[0]
    )

    base_scale = (random.random() - 0.5) / 1.5 + 1
    base_translation = (0, (random.random() - 0.5) / 2,
                        (random.random() - 0.5) / 2)

    back_mesh.apply_scale(back_scale)
    back_mesh.apply_translation(back_translation)
    back_mesh.apply_transform(back_rotation)

    base_mesh.apply_scale(base_scale)
    base_mesh.apply_translation(base_translation)


def create_chair(name):
    back_mesh = get_part_mesh(
        "Chair Back",
        chair_dir_paths[random.randint(0, len(chair_dir_paths) - 1)]
    )
    base_mesh = get_part_mesh(
        "Chair Base",
        chair_dir_paths[random.randint(0, len(chair_dir_paths) - 1)]
    )
    seat_mesh = get_part_mesh(
        "Chair Seat",
        chair_dir_paths[random.randint(0, len(chair_dir_paths) - 1)]
    )

    fix_chair_scale(back_mesh, base_mesh, seat_mesh)
    fix_chair_translation(back_mesh, base_mesh, seat_mesh)
    # break_chairs(back_mesh, base_mesh, seat_mesh)

    mesh = trimesh.util.concatenate([base_mesh, back_mesh, seat_mesh])
    mesh.export(out_dir + name + ".obj")


if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(create_chair)(str(i)) for i in tqdm(range(0, 10)))
