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
"""Type definitions used in baselines."""

from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Union

import numpy as np
import tensorflow as tf

from sdc.oatomobile import types

Shape = types.Shape
Tensor = tf.Tensor
Array = Union[types.Scalar, np.ndarray, Tensor]
NestedArray = Union[Array, Iterable["NestedArray"], Mapping[Any, "NestedArray"]]
NestedTensor = Union[Tensor, Iterable["NestedTensor"], Mapping[Any,
                                                               "NestedTensor"]]
