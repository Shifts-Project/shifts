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

from .behavioral_cloning import (
    BehaviouralModel, train_step_bc, evaluate_step_bc)
from .deep_imitative_model import (
    ImitativeModel, train_step_dim, evaluate_step_dim)
from .batch_preprocessing import batch_transform

MODEL_NAME_TO_CLASS_FNS = {
    'bc': (BehaviouralModel, train_step_bc, evaluate_step_bc),
    'dim': (ImitativeModel, train_step_dim, evaluate_step_dim)
}

MODEL_TO_FULL_NAME = {
    'bc': 'Behavioral Cloning',
    'dim': 'Deep Imitative Model'
}
