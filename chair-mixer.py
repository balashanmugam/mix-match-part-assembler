import os
import json
import trimesh
import trimesh.util
import random
from tqdm import tqdm

in_dir = "chairs/segmented/"
out_dir = "chairs/mixed/"


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


for i in tqdm(range(0, 10)):
    meshes = [
        get_part_mesh(
            "Chair Base",
            chair_dir_paths[random.randint(0, len(chair_dir_paths) - 1)]
        ),
        get_part_mesh(
            "Chair Back",
            chair_dir_paths[random.randint(0, len(chair_dir_paths) - 1)]
        ),
        get_part_mesh(
            "Chair Seat",
            chair_dir_paths[random.randint(0, len(chair_dir_paths) - 1)]
        )
    ]

    trimesh.util.concatenate(meshes).export(out_dir + str(i) + ".obj")
