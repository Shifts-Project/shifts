# Vehicle Motion Prediction Dataset
This Python package provides a API to interact with data from the Vehicle Motion Prediction task in the 2021 Shifts Challenge.

## Installation

To install the dataset and baselines training API run:
```
git clone git@github.com:yandex-research/shifts.git
cd shifts/sdc
pip install .
```

## Dataset Entry Points
The raw data is stored in the form of serialized protobuf messages. For the ```Scene``` message description refer to [Scene Protobuf](scene_protobuf.md) doc and the respective [dataset.proto](ysdc_dataset_api/proto/dataset.proto) file.

The package includes the following data loading utilities:
- A generator over raw protocol buffer data:
    - [ysdc_dataset_api.utils.scene_generator](ysdc_dataset_api/utils/reading.py#L34)
- Base class for feature producing. Provides an interface used in `MotionPredictionDataset`:
    - [ysdc_dataset_api.features.producing.FeatureProducerBase](ysdc_dataset_api/features/producing.py#L9)
- Renderer class, implementing `FeatureProducerBase` interface. Renders input scene to multichannel image according to specified config:
    - [ysdc_dataset_api.features.rendering.FeatureRenderer](ysdc_dataset_api/features/rendering.py#L468)
- PyTorch-style dataset for iterating over scenes and individual agents:
    - [ysdc_dataset_api.dataset.dataset.MotionPredictionDataset](ysdc_dataset_api/dataset/dataset.py#L23)

## Dataset Split
For the instructions on how to split datasets across different slices (location, weather conditions, etc.) and how to obtain a canonical dataset partition used in the Shifts competition refer to [**Data Partitioning**](data_partitioning.md).

## Example Notebooks
[Examples directory](examples) includes notebooks covering several topics:
- [analyze_dataset_metadata.ipynb](examples/analyze_dataset_metadata.ipynb) shows a way to analyze dataset statistics and model performance on various subsets of the data
- [compare_rip_models.ipynb](examples/compare_rip_models.ipynb) provides analysis utilities to compare baseline models across number of ensemble members and different aggregation strategies
- [data_partitioning.ipynb](examples/data_partitioning.ipynb) shows how to split dataset by scene tags
- [example.ipynb](examples/example.ipynb) describes the `FeatureRenderer` class usage, how to filter scenes using tags and iterate over the dataset
- [interacting_with_raw_data.ipynb](examples/interacting_with_raw_data.ipynb) shows a way to read raw protobuf messages and peeks inside scenes content
- [model_evaluation.ipynb](examples/model_evaluation.ipynb) illustrates a process of a model evaluation and producing a submission file for the Shifts competition

## Baselines Training
To train baseline models described in the [associated paper](https://arxiv.org/abs/2107.07455) run ```python run.py```.

For more detailed explanation refer to [Baseline Training](baselines_training.md) doc.

## Metrics
For the metrics formulation refer to [Metrics](metrics.md) doc or the [paper](https://arxiv.org/abs/2107.07455).

## Shifts Competition Submission
The submission file format is described in [Submission](submission.md) doc.
