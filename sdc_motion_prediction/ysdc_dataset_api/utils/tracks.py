import math

import numpy as np
import transforms3d as tf

from ..proto import PedestrianTrack, VehicleTrack


def get_track_polygon(track):
    position = np.array([track.position.x, track.position.y])
    yaw = track_yaw(track)
    orientation = tf.euler.euler2mat(0, 0, yaw)
    front = (orientation @ np.array([track.dimensions.x / 2, 0, 0]))[:2]
    left = (orientation @ np.array([0, track.dimensions.y / 2, 0]))[:2]
    poly = []
    poly.append(position + front + left)
    poly.append(position + front - left)
    poly.append(position - front - left)
    poly.append(position - front + left)
    poly.append(position + front + left)
    return np.array(poly)

# @profile
def get_gt_trajectory(scene, track_id):
    horizon = len(scene.future_vehicle_tracks)
    ph = np.zeros((horizon, 2), dtype=np.float32)
    for t in range(horizon):
        for track in scene.future_vehicle_tracks[t].tracks:
            if track.track_id == track_id:
                ph[t, 0] = track.position.x
                ph[t, 1] = track.position.y
    return ph


def get_tracks_polygons(tracks):
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


def track_box(track):
    return np.array([
        [track.dimensions.x, track.dimensions.y],
        [track.dimensions.x, -track.dimensions.y],
        [-track.dimensions.x, -track.dimensions.y],
        [-track.dimensions.x, track.dimensions.y],
    ]) / 2.


def track_yaw(track):
    if isinstance(track, PedestrianTrack):
        return math.atan2(track.linear_velocity.y, track.linear_velocity.x)
    elif isinstance(track, VehicleTrack):
        return track.yaw
