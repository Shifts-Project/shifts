# Submission Format

Participants are expected to submit a serialized `Submission` proto file.
The submission message includes a repeated field for predictions. An `ObjectPrediction` message should be created to correspond with each of the `prediction_requests` of a given `Scene`.
The `Object Prediction` includes the following fields:
- `track_id`: id of vehicle, unique in scene
- `scene_id`: id of scene, unique in full dataset
- repeated `weighted_trajectories`:
    - `trajectory`:
        - repeated `points`: trajectory points in vehicle-centered coordinate system. Note that only x and y coordinates are expected to be predicted.
    - `weight`: positive trajectory weight, e.g. confidence of the predicted trajectory in multi-modal prediction. Higher weight correspond to more likely trajectory. Used to aggregate displacement metrics across modes.
- `uncertainty_measure`: predicted scene-level uncertainty. Note that for our baseline, the uncertainty_measure is the negation of the per--prediction request confidence score.
- `is_ood`: boolean indicating if the prediction request corresponds to the OOD dataset (`is_ood`=True) or in-domain dataset (`is_ood`=False)