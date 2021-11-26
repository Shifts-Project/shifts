# Scene Protobuf
For protobuf message definitions refer to [proto directory](ysdc_dataset_api/proto).

Each file inside a protobuf dataset directory contains a serialized protobuf message `Scene`.

Each `Scene` message includes the following fields:
- `id`: string, unique scene id
- `past_vehicle_tracks`: repeated field containing 25 historical vehicle tracks
    - `tracks`: repeated field containing a `VehicleTrack` for all vehicles seen at current time
        - `track_id`: id of the vehicle, unique in scene
        - `position`: 3D position vector
        - `dimensions`: 3D vector of bounding box sizes
        - `linear_velocity`: 3D velocity vector, with z=0
        - `linear_acceleration`: 3D acceleration vector, with z=0
        - `yaw`: vehicle orientation
- `past_pedestrian_tracks`: repeated field containing 25 historical pedestrian tracks
    - `tracks`: repeated field containing PedestrianTrack for all vehicles seen at current time
        - `track_id`: id of the pedestrian, unique in scene
        - `position`: 3D position vector
        - `dimensions`: 3D vector of bounding box sizes
        - `linear_velocity`: 3D velocity vector, with z=0
- `past_ego_track`: repeated field containing 25 historical tracks of self-driving vehicle
- `future_vehicle_tracks`: repeated field containing 25 future vehicle tracks, used as ground truth trajectories
- `prediction_requests`: repeated field containing vehicle `track_id` and its associated trajectory tags
    - `track_id`: id of the vehicle, unique in scene
    - `trajectory_tags`: repeated field with trajectory tags
- `path_graph`: HD map information, such as lane centers, lane speed limits, lane priorities, crosswalk polygons, road polygons
- `traffic_lights`: repeated field with traffic light ids and states
- `scene_tags`: tags characterizing current scene in different aspects, such as time of the day, season, track (city where the scene was recorded), precipitation type if any.