import numba
import numpy as np
import transforms3d as tf

from ..proto import Scene, VehicleTrack


def get_latest_track_state_by_id(scene: Scene, track_id: int) -> VehicleTrack:
    """Returns a VehicleTrack instance with corresponding track_id at the last history timestamp.
    If track with track_id is not present at the last history timestamp
    then it is linearly interpolated to this timestamp from the closest past state
    and the first future state.

    Args:
        scene (Scene): scene to look for the track in
        track_id (int): track_id to look for

    Returns:
        VehicleTrack: Resulting track instance.
    """
    track, offset = _get_last_known_track_state_and_offset(scene, track_id)
    if offset < -1:
        gt_track = _get_first_gt_track_value(scene, track_id)
        ratio = -offset / (1 - offset)
        track = _linear_interpolate_vehicle_track(track, gt_track, ratio)
    return track


def _get_last_known_track_state_and_offset(scene, track_id):
    for i in range(-1, -len(scene.past_vehicle_tracks), -1):
        trackid2track = {t.track_id: t for t in scene.past_vehicle_tracks[i].tracks}
        if track_id in trackid2track:
            return trackid2track[track_id], i
    raise ValueError(
        f'past track track_id {track_id} was not found in scene {scene.id}')


def _get_first_gt_track_value(scene, track_id):
    for track in scene.future_vehicle_tracks[0].tracks:
        if track.track_id == track_id:
            return track
    raise ValueError(
        f'future track for track_id {track_id} was not found in scene {scene.id}')


def get_to_track_frame_transform(track: VehicleTrack) -> np.ndarray:
    """Produces transform from global coordinates to actor-centric coordinate system.
    So that origin is located at the vehicle center and the vehicle is headed towards positive
    direction of x axis.

    Args:
        track (VehicleTrack): Vehicle track to get transform for.

    Returns:
        np.ndarray: transformation matrix, shape (4, 4).
    """
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
def transform_2d_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transforms a bunch of 2D points stored in numpy array with transform matrix.

    Args:
        points (np.ndarray): array, shape (N, 2)
        transform (np.ndarray): transformation matrix, shape (4, 4)

    Returns:
        np.ndarray: transformed points, shape (N, 2)
    """
    ph = np.zeros((4, points.shape[0]), dtype=np.float32)
    ph[:2, :] = points.transpose()
    ph[3, :] = np.ones(points.shape[0])
    res = transform @ ph
    return np.transpose(res[:2, :])


@numba.jit(numba.float32[:, :](numba.float32[:, :], numba.float32[:, :]), nopython=True)
def transform_2d_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transforms a bunch of 2D vectors stored in numpy array with transform matrix.

    Args:
        vectors (np.ndarray): array, shape (N, 2)
        transform (np.ndarray): transformation matrix, shape (4, 4)

    Returns:
        np.ndarray: transformed vectors, shape (N, 2)
    """
    vectors = transform_2d_points(vectors, transform)
    vectors = vectors - np.asarray([[transform[0, 3], transform[1, 3]]])
    return vectors


def _linear_interpolate_vehicle_track(first, second, ratio):
    track = VehicleTrack()
    track.track_id = first.track_id
    track.dimensions.x = first.dimensions.x
    track.dimensions.y = first.dimensions.y
    track.dimensions.z = first.dimensions.z

    track.position.x = _linear_interpolate(first.position.x, second.position.x, ratio)
    track.position.y = _linear_interpolate(first.position.y, second.position.y, ratio)
    track.position.z = _linear_interpolate(first.position.z, second.position.z, ratio)

    track.linear_velocity.x = _linear_interpolate(
        first.linear_velocity.x, second.linear_velocity.x, ratio)
    track.linear_velocity.y = _linear_interpolate(
        first.linear_velocity.y, second.linear_velocity.y, ratio)
    track.linear_velocity.z = _linear_interpolate(
        first.linear_velocity.z, second.linear_velocity.z, ratio)

    track.linear_acceleration.x = _linear_interpolate(
        first.linear_acceleration.x, second.linear_acceleration.x, ratio)
    track.linear_acceleration.y = _linear_interpolate(
        first.linear_acceleration.y, second.linear_acceleration.y, ratio)
    track.linear_acceleration.z = _linear_interpolate(
        first.linear_acceleration.z, second.linear_acceleration.z, ratio)

    track.yaw = _linear_interpolate(first.yaw, second.yaw, ratio)
    return track


def _linear_interpolate(v1, v2, ratio):
    return v1 + (v2 - v1) * ratio
