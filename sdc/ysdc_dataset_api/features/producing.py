from typing import Dict, Union

import numpy as np
import torch

from ..proto import PredictionRequest, Scene


class FeatureProducerBase:
    """Base class declaring feature producing interface used in MotionPredcitionDataset."""
    def produce_features(
            self, scene: Scene, request: PredictionRequest, *args, **kwargs
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        raise NotImplementedError()

    def get_tf_signature(self):
        raise NotImplementedError()
