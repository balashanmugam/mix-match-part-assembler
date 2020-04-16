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

in_dir = "bad-chairs/input/"
out_dir = "bad-chairs/output/objs/"


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

def fix_chair_transforms(back_mesh,base_mesh,seat_mesh):
    # Allign X and Z coordinates of base to seat
    base_mesh_middle_point = ((base_mesh.bounds[0] + base_mesh.bounds[1]) / 2)
    seat_mesh_middle_point = ((seat_mesh.bounds[0] + seat_mesh.bounds[1]) / 2)
    dist_middle_points = seat_mesh_middle_point - base_mesh_middle_point

    # Delta Y
    base_translation = (dist_middle_points[0], - base_mesh.bounds[1][1] + seat_mesh.bounds[0][1] + 0.04, dist_middle_points[2])
    base_mesh.apply_translation(base_translation)

    # Lower back part
    back_translation = 0, - back_mesh.bounds[0][1] + seat_mesh.bounds[1][1] - 0.04, 0
    back_mesh.apply_translation(back_translation)

def fix_chair_scale(back_mesh, base_mesh, seat_mesh):
    delta_xz_seat = seat_mesh.bounds[1] - seat_mesh.bounds[0]
    delta_xz_base = base_mesh.bounds[1] - base_mesh.bounds[0]
    scale = delta_xz_seat / delta_xz_base

    scale_translation_matrix = np.array([
        [scale[0], 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, scale[2], 0],
        [0, 0, 0, 1]
    ])
    base_mesh.apply_transform(scale_translation_matrix)

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
    #fix_chair_transforms(back_mesh,base_mesh,seat_mesh)
    #fix_chair_scale(back_mesh,base_mesh,seat_mesh)
    break_chairs(back_mesh,base_mesh,seat_mesh)

    mesh = trimesh.util.concatenate([base_mesh, back_mesh, seat_mesh])
    mesh.export(out_dir + name + ".obj")


if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(create_chair)(str(i)) for i in tqdm(range(0, 10)))
