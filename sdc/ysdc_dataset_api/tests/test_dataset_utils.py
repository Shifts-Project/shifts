from ..dataset import data_generator
from ..dataset.utils import _trivial_filter


def test_data_generator(scene_file_path):
    for data_item in data_generator(
            [scene_file_path], [], None, True, _trivial_filter, False, True):
        assert 'ground_truth_trajectory' in data_item
