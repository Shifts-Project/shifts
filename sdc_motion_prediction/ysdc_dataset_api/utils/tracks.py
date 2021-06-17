import math
from typing import List, Union

import numpy as np
import transforms3d as tf

from ..proto import PedestrianTrack, Scene, VehicleTrack


def get_gt_trajectory(scene: Scene, track_id: int) -> np.ndarray:
    """Extracts a ground truth trajectory from scene for an object with track_id.

    Args:
        scene (Scene): scene to extract trajectory from
        track_id (int): track id to extract trajectory for

    Returns:
        np.ndarray: array of shape (N, 2), where N is specified by dataset
    """
    horizon = len(scene.future_vehicle_tracks)
    ph = np.zeros((horizon, 2), dtype=np.float32)
    for t in range(horizon):
        for track in scene.future_vehicle_tracks[t].tracks:
            if track.track_id == track_id:
                ph[t, 0] = track.position.x
                ph[t, 1] = track.position.y
    return ph


def get_tracks_polygons(tracks: List[Union[PedestrianTrack, VehicleTrack]]) -> np.ndarray:
    """Returns a numpy array with polygons for a list of tracks.
    Each polygon represent area occupied by the respective object.
    Polygons coordinates are in global coordinate system.

    Args:
        tracks (List[Union[PedestrianTrack, VehicleTrack]]): list of tracks

    Returns:
        np.ndarray: numpy array of shape (n_tracks, 4, 2)
    """
    box_base = (np.asarray([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * 0.5)[np.newaxis, ...]
    dims = np.asarray(
        [[track.dimensions.x, track.dimensions.y] for track in tracks]
    )[:, np.newaxis, :]
    boxes = box_base * dims

    origins = np.asarray(
        [[track.position.x, track.position.y] for track in tracks]
    )[:, np.newaxis, :]

    yaws = np.asarray([track_yaw(track) for track in tracks])
    s = np.sin(yaws)
    c = np.cos(yaws)
    rotation_matrices = np.array(((c, s), (-s, c))).transpose([2, 0, 1])

    return (boxes @ rotation_matrices + origins).astype(np.float32)


def track_yaw(track: Union[PedestrianTrack, VehicleTrack]) -> float:
    """Return orientation in radians for a given track.
    Since orientation for pedestrians is not available we derive it from velocity vector.

    Args:
        track (Union[PedestrianTrack, VehicleTrack]): track to get orientation for

    Returns:
        float: orientation in radians
    """
    if isinstance(track, PedestrianTrack):
        return math.atan2(track.linear_velocity.y, track.linear_velocity.x)
    elif isinstance(track, VehicleTrack):
        return track.yaw
