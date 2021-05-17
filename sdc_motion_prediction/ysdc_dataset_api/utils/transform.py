import numba
import numpy as np
import transforms3d as tf

from ..proto import VehicleTrack


def get_track_for_transform(scene, track_id, timestamp=-1):
    track, offset = get_last_track_offset(scene, track_id)
    if offset < -1:
        gt_track = get_first_gt_track_value(scene, track_id)
        ratio = -offset / (1 - offset)
        track = linear_interpolate_vehicle_track(track, gt_track, ratio)
    return track


def get_last_track_offset(scene, track_id):
    for i in range(-1, -len(scene.past_vehicle_tracks), -1):
        trackid2track = {t.track_id: t for t in scene.past_vehicle_tracks[i].tracks}
        if track_id in trackid2track:
            return trackid2track[track_id], i
    raise ValueError(
        f'past track track_id {track_id} was not found in scene {scene.id}')


def get_first_gt_track_value(scene, track_id):
    for track in scene.future_vehicle_tracks[0].tracks:
        if track.track_id == track_id:
            return track
    raise ValueError(
        f'future track for track_id {track_id} was not found in scene {scene.id}')


def get_to_track_frame_transform(track):
    position = np.array([
        track.position.x,
        track.position.y,
        0,
    ])
    yaw = track.yaw

    # Flip orientation so that non-zero velocity is directed forwards
    orientation_vec = np.array([np.cos(yaw), np.sin(yaw)])
    velocity_vec = np.array([track.linear_velocity.x, track.linear_velocity.y])
    if velocity_vec @ orientation_vec < 0:
        yaw += np.pi

    rotation = tf.euler.euler2mat(0, 0, yaw)
    transform = tf.affines.compose(position, rotation, np.ones(3))
    transform = np.linalg.inv(transform).astype(np.float32)
    return transform


@numba.jit(numba.float32[:, :](numba.float32[:, :], numba.float32[:, :]), nopython=True)
def transform2dpoints(points, transform):
    ph = np.zeros((points.shape[0], 4), dtype=np.float32)
    ph[:, :2] = points
    ph[:, 3] = np.ones(points.shape[0])
    ph = np.transpose(ph)
    res = np.dot(transform, ph)
    return np.transpose(res[:2, :])


def linear_interpolate_vehicle_track(first, second, ratio):
    track = VehicleTrack()
    track.track_id = first.track_id
    track.dimensions.x = first.dimensions.x
    track.dimensions.y = first.dimensions.y
    track.dimensions.z = first.dimensions.z

    track.position.x = interpolate(first.position.x, second.position.x, ratio)
    track.position.y = interpolate(first.position.y, second.position.y, ratio)
    track.position.z = interpolate(first.position.z, second.position.z, ratio)

    track.linear_velocity.x = interpolate(
        first.linear_velocity.x, second.linear_velocity.x, ratio)
    track.linear_velocity.y = interpolate(
        first.linear_velocity.y, second.linear_velocity.y, ratio)
    track.linear_velocity.z = interpolate(
        first.linear_velocity.z, second.linear_velocity.z, ratio)

    track.linear_acceleration.x = interpolate(
        first.linear_acceleration.x, second.linear_acceleration.x, ratio)
    track.linear_acceleration.y = interpolate(
        first.linear_acceleration.y, second.linear_acceleration.y, ratio)
    track.linear_acceleration.z = interpolate(
        first.linear_acceleration.z, second.linear_acceleration.z, ratio)

    track.yaw = interpolate(first.yaw, second.yaw, ratio)
    return track


def interpolate(v1, v2, ratio):
    return v1 + (v2 - v1) * ratio


def get_transformed_velocity(track, transform):
    vel_vec = np.array([track.linear_velocity.x, track.linear_velocity.y, 0, 1])
    return (transform @ vel_vec - transform[:, 3])[:2]


def get_transformed_acceleration(track, transform):
    acc_vec = np.array([track.linear_acceleration.x, track.linear_acceleration.y, 0, 1])
    return (transform @ acc_vec - transform[:, 3])[:2]
