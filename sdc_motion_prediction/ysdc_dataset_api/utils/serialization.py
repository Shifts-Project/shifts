import io
import zlib

import numpy as np


def maybe_compress(str, compress):
    return zlib.compress(str) if compress else str


def maybe_decompress(str, decompress):
    return zlib.decompress(str) if decompress else str


def serialize_numpy(arr: np.ndarray, compress: bool = False) -> str:
    """Serializes numpy array to string with optional zlib compression.

    Args:
        arr (np.ndarray): Numpy array to serialize.
        compress (bool, optional): Whether to compress resulting string with zlib or not.
            Defaults to False.

    Returns:
        str: serialized string
    """
    buf = io.BytesIO()
    assert isinstance(arr, np.ndarray)
    np.save(buf, arr)
    result = buf.getvalue()
    return maybe_compress(result, compress)


def deserialize_numpy(serialized_string: str, decompress: bool = False) -> np.ndarray:
    """Deserializes numpy array from compressed string.

    Args:
        serialized_string (str): Serialized numpy array
        decompress (bool, optional): Whether to decompress string with zlib before laoding.
            Defaults to False.

    Returns:
        np.ndarray: deserialized numpy array
    """
    str = maybe_decompress(serialized_string, decompress)
    buf = io.BytesIO(str)
    return np.load(buf)
