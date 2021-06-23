import os
from typing import Generator, List, Tuple, Union

import numpy as np

from ..proto import Scene
from .serialization import deserialize_numpy


def scenes_generator(
        file_paths: List[str], yield_fpath: bool = False
) -> Generator[Union[Tuple[Scene, str], Scene], None, None]:
    """Generates protobuf messages from files with serialized scenes.

    Args:
        file_paths (List[str]): paths to serialized protobuf scenes
        yield_fpath (bool, optional): If set to True generator yield file path as well.
          Defaults to False.

    Yields:
        Union[Scene, Tuple[Scene, str]]: scene protobuf message
          or tuple of scene protobuf message and respective file path
    """
    for fpath in file_paths:
        scene = read_scene_from_file(fpath)
        if yield_fpath:
            yield scene, fpath
        else:
            yield scene


def read_feature_map_from_file(filepath: str) -> np.ndarray:
    """Reads and deserializes pre-rendered feature map from the given file.

    Args:
        filepath (str): path to a file with serialized feature map

    Returns:
        np.ndarray: feature map
    """
    with open(filepath, 'rb') as f:
        fm = deserialize_numpy(f.read(), decompress=True)
    return fm


def read_scene_from_file(filepath: str) -> Scene:
    """Reads and deserializes one protobuf scene from the given file.

    Args:
        filepath (str): path to a file with serialized scene

    Returns:
        Scene: protobuf message
    """
    with open(filepath, 'rb') as f:
        scene_serialized = f.read()
    scene = Scene()
    scene.ParseFromString(scene_serialized)
    return scene


def get_file_paths(dataset_path: str) -> List[str]:
    """Returns a sorted list with file paths to raw protobuf files.

    Args:
        dataset_path (str): path to datset directory

    Returns:
        List[str]: sorted list of file paths
    """
    sub_dirs = sorted([
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    res = []
    for d in sub_dirs:
        # file_names = sorted(os.listdir(os.path.join(dataset_path, d)))
        file_names = sorted(os.listdir(d))
        res += [
            os.path.join(d, fname)
            for fname in file_names
            if fname.endswith('.pb')
        ]
    return res
