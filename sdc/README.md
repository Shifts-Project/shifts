# Vehicle Motion Prediction Dataset

## Dataset

### API

This Python package provides a way to interact with data from the Vehicle Motion Prediction task in the 2021 Shifts Challenge.

The package includes the following data loading utilities:
- A generator over raw protocol buffer data:
    - [ysdc_dataset_api.utils.scene_generator](ysdc_dataset_api/utils/reading.py#L6)
- Base class for feature producing. Provides an interface used in a PyTorch-style dataset:
    - [ysdc_dataset_api.features.producing.FeatureProducerBase](ysdc_dataset_api/features/producing.py#L7)
- Renderer class, implementing `FeatureProducerBase` interface. Renders input scene to multichannel image according to specified config:
    - [ysdc_dataset_api.features.rendering.FeatureRenderer](ysdc_dataset_api/features/rendering.py#L160)
- PyTorch-style dataset for iterating over scenes and individual agents:
    - [ysdc_dataset_api.dataset.dataset.MotionPredictionDataset](ysdc_dataset_api/dataset/dataset.py#L17)

For a more detailed example, take a look at the example [notebook](examples/example.ipynb).

### Installation

To install the dataset API run:
```
git clone git@github.com:yandex-research/shifts.git
cd shifts/sdc
pip install .
```

### Storage Format

The data directory should have the following enclosed components:

* Protobuf directories: `train_pb/`, `development_pb/`
* Tag files: `train_tags.txt`, `development_tags.txt`
* (If desired) rendered feature map directories: `train_rendered/`,
    `development_rendered/`
We provide rendered feature maps at 128x128 resolution with zlib compression
level 1, which can optionally be used to avoid rendering costs and significantly decrease
decompression time (demonstrated in the example [notebook](examples/example.ipynb), in the `Prerendered Dataset` section).

The expected directory structure is:
```
shifts/
   |--> sdc/
         |--> data/
               |--> train_pb/
               |--> development_pb/
               |--> train_tags.txt
               |--> development_tags.txt
               |--> train_rendered/
               |--> development_rendered/
```

For more information on where paths for various objects (such as model checkpoints, or where intermediate metrics, TensorBoard, or wandb logs will be written out), see [sdc/config.py](sdc/config.py).

### Scene Protobuf

Each file in a protobuf directory contains a serialized protobuf message `Scene`.

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

## Submission Format

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

All protobuf message definitions can be found at [ysdc_dataset_api/proto](ysdc_dataset_api/proto).

## Metrics

For a longer form description and motivation of metrics, read the competition whitepaper.
We highly recommend this, for ease of reading the LaTeX!

### Notation

Competitors are provided with a training dataset
$\mathcal{D}_\textup{train}=\{(\bm{x}_i, \bm{y}_i)\}_{i = 1}^{N}$
of time-profiled ground truth trajectories (i.e., plans) $\bm{y}$ paired with high-dimensional observations $\bm{x}$ of the corresponding scenes.
Note that we will use plans and trajectories interchangeably in the following.

$\bm{y} = (s_1, \dots, s_T)$ correspond to the trajectory of a given vehicle observed through the SDG perception stack.

Each of the $T$ states $s_t$ correspond to the x- and y-displacement of the vehicle, s.t. $\bm{y} \in \mathbb{R}^{T \times 2}$.

### Confidence Scores

- Per-Plan Confidence Scores (aka Per-Trajectory Confidence Scores)

   In our task, we expect a stochastic model to accompany its $D_i$ predicted plans on a given input $\bm{x}_i$ with scalar per-plan confidence scores $c_i^{(d)}, d \in \{1, \dots D_i\}$.
   These provide an ordering of the plausibility of the various plans predicted for a given input.
   They must be non-negative and sum to 1 (i.e., form a valid distribution).

- Per--Prediction Request Uncertainty Score

    We also expect models to produce a scalar uncertainty score U corresponding to each prediction request input $\bm{x}_i$ (this is an **uncertainty** rather than a confidence for consistency with other Shifts Challenge tasks).
    Often in the codebase we refer to a per--prediction request confidence score which is simply a negation of U; C = -U.

### Standard Metrics

- Average Displacement Error (ADE)

The average displacement error measures the quality of a predicted plan $\bm{y}$ with respect to the ground-truth plan $\bm{y}^*$ as

    \text{ADE}(\bm{y}) \coloneqq \frac{1}{T} \sum_{t = 1}^T \left\lVert s_t - s^*_t \right\rVert_2,

where $\bm{y} = (s_1, \dots, s_T)$.

- Aggregated ADE and Final Displacement Error (FDE)

Stochastic models define a predictive distribution $q(\bf{y}|\bm{x}; \bm{\theta})$, and can therefore be evaluated over the $D$ trajectories sampled for a given input $\bm{x}$.

For example, we can measure an aggregated ADE over $D$ samples with

    \text{aggADE}_D(q) \coloneqq \underset{\{\bm{y}\}_{d = 1}^{D} \sim q(\bf{y} \mid \bf{x})}{\oplus} \text{ADE}(\mathbf{y}^{d}),

where $\oplus$ is an aggregation operator, e.g., $\oplus = \min$ recovers the minimum ADE ($\text{minADE}_{D}$)

We consider minimum and mean aggregation (minADE, avgADE), as well as the final displacement error (FDE)

    \text{FDE}(\bf{y}) \coloneqq \left\lVert s_T - s^*_T \right\rVert,

as well as its aggregated variants minFDE and avgFDE.

### Per-Plan Confidence-Aware Metrics

A stochastic model used in practice for motion prediction ultimately must **decide** on a particular predicted plan for a given prediction request.
We may make this decision by selecting for evaluation the predicted plan with the **highest per-plan confidence score**.

In other words, given per-plan confidence scores $c^{(d)}, d \in \{1, \dots D\}$ we select the top plan $y^{(d^{\*})}, d^* = \underset{d}{\arg \max}\ c^{(d)}$, and measure the decision quality using ``top1'' ADE and FDE metrics, e.g.,

    \text{top1ADE}_D(q) \coloneqq \text{ADE}(\mathbf{y}^{(d^*)}).

We may also wish to assess the quality of the relative weighting of the $D$ plans with their corresponding per-plan confidence scores $c^{(d)}$. This is accomplished with a weighted metric

    \text{weightedADE}_D(q) \coloneqq \sum_{d \in D} c^{(d)} \cdot \text{ADE}(\mathbf{y}^{(d)}).

top1FDE and weightedFDE follow analogously to the above.


### Per--Prediction Request Uncertainty-Aware Metrics

We evaluate the quality of uncertainty quantification using retention-based metrics, in which the per--prediction request uncertainty scores determine retention order.

Note that each retention curve is plotted with respect to a particular metric above (e.g., we consider AUC for retention on weightedADE).

See the Performance Metrics section of the whitepaper and the [Retention Task](#retention-task) section below for more details.

## Baselines

We provide ensemble-based baselines for the Motion Prediction Task using the Robust Imitative Planning ensembling approach, and two "backbone" variants for ensemble members:
* Behavioral Cloning: MobileNetV2 encoder, GRU decoder
* Deep Imitative Model: MobileNetV2 encoder, autoregressive flow decoder

Model                                 | Paper                    |
--------------------------------------| :----------------------: |
Robust Imitative Planning (RIP)       | [Filos et al., 2020]     |
Deep Imitative Model (DIM)            | [Rhinehart et al., 2018] |
Behavoral Cloning (BC)*               | [Codevilla et al., 2017] |

*A simple backbone, based on that used in Conditional Imitation Learning (CIL, Codevilla et al. 2017).

[Codevilla et al., 2017]: https://arxiv.org/abs/1710.02410
[Rhinehart et al., 2018]: https://arxiv.org/abs/1810.06544
[Filos et al., 2020]: https://arxiv.org/abs/2006.14911

See the `Method-Specific Hypers` section in [sdc/config.py](sdc/config.py) for more information on hyperparameters specific to RIP, BC, and DIM.

### Robust Imitative Planning Overview

The RIP ensemble method ([Filos et al., 2020]) stochastically generates multiple predictions for a given prediction request.
Our adaptation of the method produces uncertainty/confidence scores at two levels of granularity:
1. Each prediction request (uncertainty score).
2. Each of the predicted plans generated for each of those prediction requests (confidence scores).

This same format is expected from competitors, as detailed in the `Submission Format` section above.

In detail, we use the following approach for plan and confidence score generation:
1. **Plan Generation.** Given a scene input in the format of a rendered image, K ensemble members generate G plans.<sup>1</sup>
2. **Plan Scoring.** We score each of the G plans by computing a log probability under each of the K trained likelihood models.
3. **Per-Plan Confidence Scores.** We aggregate the G * K total scores to G scores, using the `--rip_per_plan_algorithm` aggregating over the log-likelihood estimates sampled from the model posterior (i.e., contributed by each ensemble member) to produce a robust score for each of the G plans.
4. **Plan Selection.** Among the G plans, the RIP ensemble produces the D = `--rip_num_preds` top plans as determined by their corresponding G per-plan confidence scores.
5. **Per--Prediction Request Confidence Score.** We aggregate the D top per-trajectory confidence scores to a single score C, representing ensemble confidence for the scene context overall, using the `--rip_per_scene_algorithm`. To obtain our desired per--prediction request uncertainty score, we simply negate: U = -C.
6. **Confidence Reporting.** Competitors are expected to produce, for a given prediction request, per-plan confidence scores which are non-negative and sum to 1 (i.e., form a valid distribution). We obtain these scores $c^{(d)}$ by applying a softmax to the D top per-plan confidence scores. We report these c^{(d)} and U (computed in step 5) as our final per-plan confidence scores and per--prediction request uncertainty score, respectively.

To summarize, our implementation of RIP for motion prediction produces D plans and corresponding normalized per-plan scores c^{(d)}, as well as an aggregated uncertainty score U for the overall prediction request.
We often in the codebase refer to per--prediction request confidence scores, which correspond to the negation of U: C = -U.
### Training RIP Ensemble Members

[Lakshminarayanan et al., 2017]: https://arxiv.org/abs/1612.01474v3

RIP ensemble members are trained independently, similar to in Deep Ensembles ([Lakshminarayanan et al., 2017]).

For example, to train a Behavioral Cloning agent with no teacher forcing (which we found to have the best empirical performance among BC with/without teacher forcing and DIM) and the prerendered feature maps, we use command

```
python run.py --model_name bc --data_use_prerendered True --bc_generation_mode sampling
```

See [sdc/oatomobile/torch/baselines](sdc/oatomobile/torch/baselines) for the baseline implementations.

We can train K different ensemble members by sweeping over the `--torch_seed` parameter.

By default we checkpoint every time the ADE on the Moscow development dataset decreases. The number of improvements before a checkpoint can be specified with `--exp_checkpoint_frequency` and the loss metric/dataset can be set with `--exp_checkpoint_validation_loss`.

### Evaluating RIP with Trained Ensemble Members

Run RIP to create a directory in which you should store the ensemble member checkpoints created by the above command.

For example, with the Lower Quartile aggregation strategy for per-plan and per--prediction request confidence scores, and 5 ensemble members:

```
python run.py --model_name bc --data_use_prerendered True --bc_generation_mode sampling --torch_seed 1 --np_seed 1 --rip_per_plan_algorithm LQ --rip_per_scene_algorithm LQ --rip_k 5
```

The Torch and NumPy seeds will affect the evaluation through sampling from the model, batching, etc.

The above command will create a directory in which the RIP ensemble member checkpoints are expected.
Place them there, and re-run the command to evaluate the RIP ensemble.

See [sdc/config.py](sdc/config.py) for descriptions of all parameters.

### Performance Analysis

We provide utilities for three types of downstream analysis on RIP predictions:
1. Retention on per--prediction request uncertainty scores.
2. Dataset metadata analysis for a particular RIP method.
3. Caching all predictions and plan-level confidence scores for a complete downstream analysis of a RIP ensemble with a particular backbone (e.g., sweeping over aggregation strategies and number of ensemble members K).

Below we walk through each type of analysis, and demonstrate extra flags for evaluating RIP models which might be helpful for evaluating your own architectures.

We use Weights & Biases ([wandb](https://wandb.ai/home)) for experiment tracking.
wandb allows us to conveniently track run progress online, and can be disabled by executing `wandb offline`.

#### Retention Task

Self-driving agents that have the ability to quantify their uncertainty
in a given setting have the potential to significantly improve the safety
and success of autonomous vehicle deployment.

For example, these uncertainty estimates can be used in active learning to
designate settings in which the agent is particularly uncertain for later
exploration. Alternatively, an agent could yield control of the vehicle
to a human passenger if its uncertainty for a scene is particularly high.

We can quantify the quality of a motion prediction model's uncertainty
estimates through a `retention` task, in which the model is asked to make
predictions on a range of proportions of the development dataset.
We can assume that for the proportion of scenes that are not retained,
a human passenger is able to successfully navigate the setting (i.e.,
achieve near-perfect accuracy with respect to an expert trajectory).
Therefore, performance on the retention task can be seen as the aggregate
performance of a passenger + self-driving agent system.
These could correspond to the acceptable proportion of driving time during
which an agent could yield control to a human passenger.

A model with good uncertainty estimation will have confidence scores highly
correlated with accuracy (or negatively correlated with ADE), which will
allow it to outperform a model with the same ADE but poor uncertainty
estimation at all retention proportions < 1.

**Plotting Retention Curves**

The RIP evaluation script provided above will generate data for area under retention curve plots with various pertinent metrics, such as minADE and top1FDE (for a full account of all metrics, see [ysdc_dataset_api/evaluation/metrics.py](ysdc_dataset_api/evaluation/metrics.py)).

These are logged to wandb, and also stored as a pd.DataFrame in `{--dir_data}/metrics/{model_name}/results.tsv` (or stored under a separately specified directory `--dir_metrics`, at `{--dir_metrics}/{model_name}/results.tsv`).

You can run the RIP evaluation script multiple times with varied `--np_seed` and `--torch_seed`, and our plotting utilities will generate aggregated plots with error bars.

The command to generate retention plots for a particular model is:

```
python plot_retention_curves.py --results_dir {--dir_metrics}/{model_name} --plot_dir '.' --model_name {model_name}
```

where the `--results_dir` should point to the subdirectory of a particular model name (e.g., 'rip-dim-k_5-plan_uq-scene_uq' corresponding to a RIP ensemble with Upper Quartile aggregation, DIM backbone density models, and 5 ensemble members).

You can generate plots comparing several models with

```
python plot_retention_curves.py --results_dir {--dir_metrics} --plot_dir='.'
```

where the `--results_dir` should point to the metrics directory containing all `model_name` subdirectories.

#### Caching Dataset Metadata for Downstream Analysis

In addition to comparing performance on the retention task, you may wish to investigate how the model performs across various subsets of the data; for example, across cities, on turning/non-turning trajectories, in different weather conditions, or in low-confidence settings.

We provide utilities for this by setting the command `--debug_collect_dataset_stats=True` when running a RIP evaluation script. Metadata will be stored under the directory `--dir_metadata_cache` (or `{--dir_data}/metadata_cache` if unspecified).

As a starting point for analysis of model predictions, we provide an example notebook located at [examples/analyze_dataset_metadata.ipynb](examples/analyze_dataset_metadata.ipynb).

#### Caching All Predictions for Comparing RIP Configurations

Finally, you may wish to cache all predictions and per-plan confidence scores of constituent models in a RIP ensemble, instead of performing the RIP aggregation steps and caching losses, in order to compare RIP performance for a particular backbone (BC or DIM) across many ensemble sizes and aggregation strategies.

These utilities allow one to simply evaluate a RIP model at the largest K they wish to consider, and then compare performance with various ensemble sizes/aggregation strategies post-hoc.

In particular, the predictions, ground truth values, per-plan confidence scores, and request IDs are stored for each prediction request.
For the confidence scores, this means we store all G * K log-likelihood scorings pairwise between plans and ensemble members.
This is controlled by the `--rip_cache_all_preds=True` flag, and importantly, **should be used in conjunction with the above metadata caching** in order to retrieve the correct pairing of (request ID, scene ID) to uniquely identify every prediction request.

For example,
```
python run.py --model_name bc  --data_use_prerendered True --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 5 --debug_collect_dataset_stats True --rip_eval_subgroup eval --rip_cache_all_preds True
```
will evaluate a RIP ensemble with backbone model BC, aggregation strategy MA (for both per-plan and per--prediction request), ensemble size K = 5.

This script will produce metadata as well as cache all of the fields mentioned above.

Note also the additional flag:
`--rip_eval_subgroup eval` restricts to evaluating the RIP ensemble only on development data for convenience. You may also pass `train` to only evaluate on training data.

We provide an example [notebook](examples/compare_rip_models.ipynb) walking through downstream comparative analysis of RIP variants.

### Additional References

For additional reference papers on the Motion Prediction task, see the following:
- MultiPath: https://arxiv.org/abs/1910.05449
- CoverNet: https://arxiv.org/abs/1911.10298
- TNT: https://arxiv.org/abs/2008.08294
- LaneRCNN: https://arxiv.org/abs/2101.06653
