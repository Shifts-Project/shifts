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
"""Utility classes for logging on TensorBoard."""

import os
from typing import List
from typing import Text, Mapping

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from matplotlib import collections as mc

from sdc.dataset import load_renderer
from sdc.oatomobile.tf import types

COLORS = [
    "#0071bc",
    "#d85218",
    "#ecb01f",
    "#7d2e8d",
    "#76ab2f",
    "#4cbded",
    "#a1132e",
]


class TensorBoardLogger:
    """A simple `TensorFlow`-friendly `TensorBoard` wrapper."""

    def __init__(
        self,
        log_dir: str,
        dataset_names: List[str]
    ) -> None:
        """Constructs a simple `TensorBoard` wrapper."""
        self.dataset_names = dataset_names

        # Makes sure output directories exist.
        for dataset_name in dataset_names:
            dataset_log_dir = os.path.join(log_dir, dataset_name)
            os.makedirs(dataset_log_dir, exist_ok=True)

            # Initialises the `TensorBoard` writers.
            self.__setattr__(f'summary_writer_{dataset_name}',
                             tf.summary.create_file_writer(dataset_log_dir))

        self.renderer = load_renderer()

    def log(
        self,
        dataset_name: str,
        loss_dict: Mapping[Text, torch.Tensor],
        overhead_features: types.Array,
        predictions: types.Array,
        ground_truth: types.Array,
        global_step: int,
    ) -> None:
        """Logs the scalar loss and visualizes predictions for qualitative
        inspection.

        Args:
          dataset_name: One of the dataset names used in init,
            raises error otherwise.
          loss_dict: The optimization objective values.
          overhead_features: The overhead visual input  with shape `[B, C, H, W]`.
          predictions: Samples from the model, with shape `[B, (K,) T, 2]`.
          ground_truth: The ground truth plan, with shape `[B, T, 2]`.
          global_step: The x-axis value.
        """

        if dataset_name in self.dataset_names:
            summary_writer = getattr(
                self, f'summary_writer_{dataset_name}', None)
            if summary_writer is None:
                raise ValueError(f"Could not locate summary writer for "
                                 f"dataset {dataset_name}")
        else:
            raise ValueError(f'Unrecognized dataset name '
                             f'{dataset_name} passed.')

        with summary_writer.as_default():
            for loss_name, loss in loss_dict.items():
                loss = loss.cpu().detach().numpy().item()
                tf.summary.scalar(loss_name, data=loss, step=global_step)

            # Visualizes the predictions.
            raw = list()
            for _, (
                o_t,
                p_t,
                g_t,
            ) in enumerate(zip(
                overhead_features,
                predictions,
                ground_truth,
            )):
                fig = plt.figure(figsize=(10, 10))
                hw_size = o_t.shape[1]
                p_t = p_t + (hw_size / 2)
                g_t = g_t + (hw_size / 2)
                p_t = np.transpose(p_t)
                g_t = np.transpose(g_t)
                p_t = p_t[::-1]
                g_t = g_t[::-1]
                p_t = np.transpose(p_t)
                g_t = np.transpose(g_t)
                plt.imshow(o_t[0], origin='lower', cmap='binary', alpha=0.1)
                plt.imshow(o_t[6], origin='lower', cmap='binary', alpha=0.1)
                plt.imshow(o_t[12], origin='lower', cmap='binary', alpha=0.1)
                plt.imshow(o_t[15], origin='lower', cmap='binary', alpha=0.1)
                ax = plt.gca()
                ax.add_collection(
                    mc.LineCollection([g_t], color='green'))
                ax.add_collection(
                    mc.LineCollection([p_t], color='blue'))
                ax.legend()
                ax.set(frame_on=False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Convert `matplotlib` canvas to `NumPy` with 4 channels.
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf.shape = (w, h, 4)
                buf = np.roll(buf, 3, axis=2)
                raw.append(buf)
                plt.close(fig)
            raw = np.reshape(np.asarray(raw), (-1, w, h, 4))
            tf.summary.image("examples", raw, max_outputs=16, step=global_step)
