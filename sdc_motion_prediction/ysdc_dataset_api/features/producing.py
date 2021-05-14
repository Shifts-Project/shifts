from typing import Union, Dict

import numpy as np
import torch


class FeatureProducerBase:
    def produce_features(
            self, scene, *args, **kwargs
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        raise NotImplementedError()
