import numpy as np
import transforms3d as tf


def tracks_as_np():
    pass

def get_track_polygon(track):
    position = np.array([track.position.x, track.position.y])
    orientation = tf.euler.euler2mat(0, 0, track.yaw)
    front = (orientation @ np.array([track.dimensions.x / 2, 0, 0]))[:2]
    left = (orientation @ np.array([0, track.dimensions.y / 2, 0]))[:2]
    poly = []
    poly.append(position + front + left)
    poly.append(position + front - left)
    poly.append(position - front - left)
    poly.append(position - front + left)
    poly.append(position + front + left)
    return np.array(poly)
