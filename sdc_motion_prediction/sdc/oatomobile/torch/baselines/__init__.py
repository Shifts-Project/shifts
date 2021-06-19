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

from pprint import pprint

from sdc.oatomobile.torch.baselines.robust_imitative_planning import (
    evaluate_step_rip, RIPAgent)
from sdc.oatomobile.torch.baselines.batch_preprocessing import batch_transform
from sdc.oatomobile.torch.baselines.behavioral_cloning import (
    BehaviouralModel, train_step_bc, evaluate_step_bc)
from sdc.oatomobile.torch.baselines.deep_imitative_model import (
    ImitativeModel, train_step_dim, evaluate_step_dim)


def init_behavioral_model(c):
    kwargs = get_bc_kwargs(c)
    print('Model kwargs:')
    pprint(kwargs)
    return BehaviouralModel(**kwargs).to(kwargs['device'])


def get_bc_kwargs(c):
    return {
        'in_channels': c.model_in_channels,
        'dim_hidden': c.model_dim_hidden,
        'output_shape': c.model_output_shape,
        'device': c.exp_device,
        'debug_bc_deterministic': c.debug_bc_deterministic
    }


def init_imitative_model(c):
    kwargs = get_dim_kwargs(c)
    print('Model kwargs:')
    pprint(kwargs)
    return ImitativeModel(**kwargs).to(kwargs['device'])


def get_dim_kwargs(c):
    return {
        'in_channels': c.model_in_channels,
        'dim_hidden': c.model_dim_hidden,
        'output_shape': c.model_output_shape,
        'device': c.exp_device,
        'scale_eps': c.dim_scale_eps
    }


def init_rip(c):
    # Init kwargs/config items
    ensemble_kwargs = get_rip_kwargs(c)
    k = ensemble_kwargs['k']
    print('RIP kwargs:')
    pprint(ensemble_kwargs)
    per_plan_algorithm = c.rip_per_plan_algorithm
    per_scene_algorithm = c.rip_per_scene_algorithm
    model_name = c.model_name

    print(f'Building RIP agent with backbone model {model_name}, '
          f'per-plan algorithm {per_plan_algorithm}, '
          f'per-scene algorithm {per_scene_algorithm}, '
          f'{k} ensemble members.')
    full_model_name = (
        f'rip-{model_name}-k_{k}-plan_{per_plan_algorithm}-scene'
        f'_{per_scene_algorithm}').lower()

    # Init models
    backbone_init_fn, _, _ = BACKBONE_NAME_TO_CLASS_FNS[model_name]
    models = [backbone_init_fn(c) for _ in range(k)]
    return (RIPAgent(models=models, **ensemble_kwargs), full_model_name,
            evaluate_step_rip, evaluate_step_rip)


def init_model(c):
    model_name = c.model_name
    per_plan_algorithm = c.rip_per_plan_algorithm
    per_scene_algorithm = c.rip_per_scene_algorithm
    if per_plan_algorithm is None or per_scene_algorithm is None:
        print(f'Training {BACKBONE_NAME_TO_FULL_NAME[model_name]}')
        init_fn, train_step, test_step = (
            BACKBONE_NAME_TO_CLASS_FNS[model_name])
        model = init_fn(c)
        return model, model_name, train_step, test_step
    else:
        return init_rip(c)


def get_rip_kwargs(c):
    return {
        'per_plan_algorithm': c.rip_per_plan_algorithm,
        'per_scene_algorithm': c.rip_per_scene_algorithm,
        'k': c.rip_k,
        'model_name': c.model_name,
        'device': c.exp_device,
        'samples_per_model': c.rip_samples_per_model,
        'num_preds': c.rip_num_preds
    }


BACKBONE_NAME_TO_KWARGS_FN = {
    'bc': get_bc_kwargs,
    'dim': get_dim_kwargs
}

BACKBONE_NAME_TO_CLASS_FNS = {
    'bc': (init_behavioral_model, train_step_bc, evaluate_step_bc),
    'dim': (init_imitative_model, train_step_dim, evaluate_step_dim),
    # TODO: RIP wrapper on BC (rnn decoder)
    # TODO: RIP with MC Dropout?
}

BACKBONE_NAME_TO_FULL_NAME = {
    'bc': 'Behavioral Cloning',
    'dim': 'Deep Imitative Model'
}
