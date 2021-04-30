from typing import Union

import numpy as np
import torch


class FeatureProducerBase:
    def produce_features(
            self, scene, *args, **kwargs
    ) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        raise NotImplementedError()
