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
"""Core data structures and type definitions."""

from typing import Sequence
from typing import Union

import numpy as np


class Singleton(type):
  """Implements the singleton pattern."""

  _instances = {}

  def __call__(cls, *args, **kwargs):
    """Checks if singleton exists, creates one if not else return it."""
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]


Shape = Sequence[int]
Shape = Union[int, Shape]
Scalar = Union[float, int]
