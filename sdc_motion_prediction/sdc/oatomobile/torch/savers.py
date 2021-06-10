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
"""Utility classes for saving model checkpoints."""

import os

import torch
import torch.nn as nn

# Determines device, accelerator.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member


class Checkpointer:
  """A simple `PyTorch` model load/save wrapper."""
  def __init__(
      self,
      model: nn.Module,
      torch_seed: int,
      ckpt_dir: str,
      checkpoint_frequency: int
  ) -> None:
    """Constructs a simple load/save checkpointer."""
    self._model = model
    self._torch_seed = torch_seed
    self._ckpt_dir = ckpt_dir
    self._best_validation_loss = float('inf')

    # The number of times validation loss must improve prior to our
    # checkpointing of the model
    if checkpoint_frequency == -1:
      self.checkpoint_frequency = float('inf')  # We will never checkpoint
    else:
      self.checkpoint_frequency = checkpoint_frequency

    self._num_improvements_since_last_ckpt = 0

    os.makedirs(self._ckpt_dir, exist_ok=True)

  def save(
      self,
      epoch: int,
      new_validation_loss: float
  ) -> str:
    """Saves the model to the `ckpt_dir/epoch/model.pt` file."""
    model_file_name = f'model-seed-{self._torch_seed}-epoch-{epoch}.pt'

    if new_validation_loss < self._best_validation_loss:
      self._num_improvements_since_last_ckpt += 1
      self._best_validation_loss = new_validation_loss

      if self._num_improvements_since_last_ckpt >= self.checkpoint_frequency:
        print(
          f'Validation loss has improved '
          f'{self._num_improvements_since_last_ckpt} times since last ckpt, '
          f'storing checkpoint {model_file_name}.')
        ckpt_path = os.path.join(self._ckpt_dir, model_file_name)
        torch.save(self._model.state_dict(), ckpt_path)
        self._num_improvements_since_last_ckpt = 0
        return ckpt_path

  def load(
      self,
      epoch: int,
  ) -> nn.Module:
    """Loads the model from the `ckpt_dir/epoch/model.pt` file."""
    model_file_name = f'model-seed-{self._torch_seed}-epoch-{epoch}.pt'
    ckpt_path = os.path.join(self._ckpt_dir, model_file_name)
    self._model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return self._model
