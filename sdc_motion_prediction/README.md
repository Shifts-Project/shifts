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

All protobuf message definitions can be found at [ysdc_dataset_api/proto](ysdc_dataset_api/proto)
## Reference papers for motion prediction task
- MultiPath: https://arxiv.org/abs/1910.05449
- CoverNet: https://arxiv.org/abs/1911.10298
- TNT: https://arxiv.org/abs/2008.08294
- LaneRCNN: https://arxiv.org/abs/2101.06653