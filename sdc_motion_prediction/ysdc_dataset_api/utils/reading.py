import os

from ..proto import Scene


def scenes_generator(file_paths):
    for fpath in file_paths:
        scene = read_scene_from_file(fpath)
        yield scene


def read_scene_from_file(filepath):
    with open(filepath, 'rb') as f:
        scene_serialized = f.read()
    scene = Scene()
    scene.ParseFromString(scene_serialized)
    return scene


def get_file_paths(dataset_path):
    sub_dirs = sorted([
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    res = []
    for d in sub_dirs:
        file_names = sorted(os.listdir(d))
        res += [
            os.path.join(d, fname)
            for fname in file_names
            if fname.endswith('.pb')
        ]
    return res
