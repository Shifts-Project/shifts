import io
import zlib

import numpy as np


def maybe_compress(str, compress):
    return zlib.compress(str) if compress else str


def maybe_decompress(str, decompress):
    return zlib.decompress(str) if decompress else str


def serialize_numpy(arr, compress=False):
    buf = io.BytesIO()
    assert isinstance(arr, np.ndarray)
    np.save(buf, arr)
    result = buf.getvalue()
    return maybe_compress(result, compress)


# @profile
def deserialize_numpy(str, decompress=False):
    str = maybe_decompress(str, decompress)
    buf = io.BytesIO(str)
    return np.load(buf)
