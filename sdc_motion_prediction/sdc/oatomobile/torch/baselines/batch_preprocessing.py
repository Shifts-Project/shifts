# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Per-batch preprocessing/downsampling."""

from typing import Mapping
from typing import Tuple

import torch

from sdc.dataset import torch_cast_to_dtype
from sdc.oatomobile.torch import transforms


def batch_transform(
    sample,
    device: str = 'cpu',
    downsample_hw: Tuple[int, int] = (100, 100),
    dtype: str = 'float32',
    num_timesteps_to_keep: int = 25
) -> Mapping[str, torch.Tensor]:
    """Prepares variables for the interface of the model.

    Args:
      sample: (keyword arguments) The raw sample variables.

    Returns:
      The processed sample.
    """
    if "ground_truth_trajectory" in sample.keys():
        player_future = sample["ground_truth_trajectory"]
        _, T, _ = player_future.shape
        if T != num_timesteps_to_keep:
            sample["ground_truth_trajectory"] = transforms.downsample_target(
                player_future=player_future,
                num_timesteps_to_keep=num_timesteps_to_keep,
            )

        sample["ground_truth_trajectory"] = (
            torch_cast_to_dtype(
                sample["ground_truth_trajectory"], dtype).to(device=device))

    # Preprocesses the visual features.
    if "feature_maps" in sample.keys():
        sample["feature_maps"] = torch_cast_to_dtype(
                transforms.transpose_visual_features(
                    transforms.downsample_visual_features(
                        visual_features=sample["feature_maps"],
                        output_shape=downsample_hw,
                    )), dtype).to(device=device)

    return sample
