# Vehicle motion prediction dataset
## API
This python package provides a way to interact with vehicle motion prediction data of the respective Shifts Challenge track.

Package includes following utilities:
- Generator over raw protocol buffer data:
    - [ysdc_dataset_api.utils.scene_generator](ysdc_dataset_api/utils/reading.py#L6)
- Base class for feature producing. Provides interface used inside pytorch style dataset:
    - [ysdc_dataset_api.features.producing.FeatureProducerBase](ysdc_dataset_api/features/producing.py#L7)
- Renderer class, implementing FeatureProducerBase interface. Renders input scene to multichannel image according to specified config:
    - [ysdc_dataset_api.features.rendering.FeatureRenderer](ysdc_dataset_api/features/rendering.py#L160)
- Pytorch-style dataset for iterating over scenes and individual agents:
    - [ysdc_dataset_api.dataset.dataset.MotionPredictionDataset](ysdc_dataset_api/dataset/dataset.py#L17)

For more detailed example take a look at the example [notebook](examples/example.ipynb).
## Installation
To install dataset api run:
```
git clone git@github.com:yandex-research/uncertainty-challenge.git
cd uncertainty-challenge/sdc_motion_prediction
pip install .
```
## Raw data format description
Each file in a dataset directory contains a serialized protobuf message Scene.
Scene message includes the following fields:
- id: string, unique scene id
- past_vehicle_tracks: repeated field containing 25 historical vehicle tracks
    - tracks: repeated field containing VehicleTrack for all vehicles seen at current time
        - track_id: unique vehicles id in scene
        - position: 3d position vector
        - dimensions: 3d vector of bounding box sizes
        - linear_velocity: 3d velocity vector, with z=0
        - linear_acceleration: 3d acceleration vector, with z=0
        - yaw: vehicle orientation
- past_pedestrian_tracks: repeated field containing 25 historical pedestrian tracks
    - tracks: repeated field containing PedestrianTrack for all vehicles seen at current time
        - track_id: unique pedestrian id in scene
        - position: 3d position vector
        - dimensions: 3d vector of bounding box sizes
        - linear_velocity: 3d velocity vector, with z=0
- past_ego_track: repeated field containing 25 historical tracks of self-driving vehicle
- future_vehicle_tracks: repeated field containing 25 future vehicle tracks, used as ground truth trajectories
- prediction_requests: repeated field containing vehicle track_id and its associated trajectory tags
    - track_id: unique vehicle id in scene
    - trajectory_tags: repeated field with trajectory tags
- path_graph: HD map information, such as lane centers, lane speed limits, lane priorities, crosswalk polygons, road polygons
- traffic_lights: repeated field with traffic light ids and states
- scene_tags: tags characterizing current scene in different aspects, such as time of the day, season, track (city where the scene was recorded), precipitation type if any.

## Submission format
Participants are expected to submit a serialized Submission proto file.
Submission message includes repeated field predictions. Each ObjectPrediction messages includes fields:
- track_id: unique vehicle id in scene
- scene_id: unique scene id in dataset
- repeated weighted_trajectories:
    - trajectory:
        - repeated points: trajectory points in vehicle-centered coordinate system. Note that only x and y coordinates are expected to be predicted.
    - weight: positive trajectory weight, e.g. confidence of the predicted trajectory in multi-modal prediction. Higher weight correspond to more likely trajectory. Used to aggregate displacement metrics across modes.
- uncertainty_measure: predicted scene-level confidence

All protobuf message definitions can be found at [ysdc_dataset_api/proto](ysdc_dataset_api/proto)
## Reference papers for motion prediction task
- MultiPath: https://arxiv.org/abs/1910.05449
- CoverNet: https://arxiv.org/abs/1911.10298
- TNT: https://arxiv.org/abs/2008.08294
- LaneRCNN: https://arxiv.org/abs/2101.06653
<<<<<<< HEAD

## Robust Imitative Planning baseline

In Robust Imitative Planning, we use an ensemble of Deep Imitative Models (DIM, \cite{}) to produce confidence scores at (a) the level of scenes, and (b) at the level of plans generated for each of those scenes.

We use the following approach to obtain these scores and stochastically generate plan predictions:

1. Given a scene input, K ensemble members generate D plans (in practice, we have each ensemble member generate the same number of plans, controlled by Q = `--rip_samples_per_model`, s.t. D = K * Q).
2. 



* `--rip_num_preds`: number of plan predictions that are actually made by the RIP model for a given prediction request input. 



which are used to score D generated plans.

We obtain per-plan uncertainty scores 

## Training RIP ensemble members

RIP ensemble members are trained independently, similar to in Deep Ensembles \cite{balaji}

Here we use the Deep Imitative Model as a backbone density estimator and sweep over x \in {1, ..., K} different seeds:

```
python run.py --model_name dim --data_use_prerendered True --torch_seed x
```

## Evaluating RIP with trained ensemble members

Run RIP to create a directory in which you should store the ensemble member checkpoints created by the above command.

For example, with the Worst Case Model algorithm aggregating  and 5 ensemble members.

Here, the torch seed will affect the batching used at evaluation time. 
# TODO: nband -> confirm that it actually does change things if we are not shuffling?

```
python run.py --model_name dim --data_use_prerendered True --torch_seed 1 --rip_algorithm WCM --rip_k 5
```

The above command will create a directory in which the RIP ensemble member checkpoints are expected.
Place them there, and re-run the command to evaluate the RIP ensemble.

Other parameters in RIP include:
We select these as the top samples from all stochastic generations from the ensemble members.

See `sdc/config.py` for descriptions of all parameters.


## Plotting Retention Curves

This evaluation will generate data for area under retention curve plots with various pertinent metrics, such as minADE and minFDE.

These are logged to wandb, and also stored as a pd.DataFrame in `{--dir_data}/metrics/{model_name}/results.tsv`.

We use these to generate retention curve plots. 

You can generate plots for a particular model with

```
python plot_retention_curves.py --results_dir {--dir_metrics}/{model_name} --plot_dir '.' --model_name {model_name}
```

where the --results_dir should point to the subdirectory of a particular model name (e.g., 'rip-wcm-dim-k_5' corresponding to a RIP ensemble with the WCM algorithm, DIM backbone density models, and 5 ensemble members).

You can generate plots comparing several models with

```
python plot_retention_curves.py --results_dir {--dir_metrics} --plot_dir='.'
```

where the --results_dir should point to the metrics directory containing all model_name subdirectories.
