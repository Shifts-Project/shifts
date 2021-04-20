import numpy as np
import transforms3d as tf


def get_track_for_transform(scene, track_id, timestamp=-1):
    track = None
    for t in scene.past_vehicle_tracks[-1].tracks:
        if t.track_id == track_id:
            track = t
    if track is None:
        raise ValueError('Track with track_id {} was not found in scene'.format(track_id))
    return track

def get_track_to_fm_transform(track):
    position = np.array([
        track.position.x,
        track.position.y,
        track.position.z
    ])
    yaw = track.yaw
    orientation_vec = np.array([np.cos(yaw), np.sin(yaw)])
    # Flip orientation so that non-zero velocity is directed forwards
    if position[:2] @ orientation_vec < 0:
        yaw += np.pi
    rotation = tf.euler.euler2mat(0, 0, yaw)
    transform = tf.affines.compose(position, rotation, np.ones(3))
    transform = np.linalg.inv(transform)
    return transform
