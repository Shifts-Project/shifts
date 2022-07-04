import pandas as pd
from typing import Union, Dict

GLOBAL_TIME_MAXIMA = {
    "timeSinceDryDock": 4324320
}


class FeatureScaler(object):
    def __init__(self, config: Dict[str, str]):
        """
        A class for handling the normalization of the features.
        :param config: A dictionary containing a mapping for each feature to its type of scaling
        """
        super(FeatureScaler, self).__init__()

        self.config = config
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, data: pd.DataFrame) -> None:
        self.mean = data.mean()
        self.std = data.std()
        self.min = data.min()
        self.max = data.max()

    def series_transform(self, series, norm_type, feature):
        if norm_type == 'zeta':
            norm_data = (series - self.mean[feature]) / self.std[feature]
        elif norm_type == 'time':
            norm_data = series / GLOBAL_TIME_MAXIMA[feature]
        elif norm_type == 'none':
            norm_data = series
        else:
            raise Exception(f"Not supported type of normalization '{norm_type}' for feature '{feature}'. ")
        return norm_data

    def transform(self, data: Union[pd.DataFrame, pd.Series], feature=None) -> pd.DataFrame:
        if feature is None:
            norm_data: pd.DataFrame = data.copy()
            for feature in data.columns:
                if feature in self.config:
                    norm_type = self.config[feature]
                    norm_data[feature] = self.series_transform(series=data[feature], norm_type=norm_type,
                                                               feature=feature)
        else:
            norm_type = self.config[feature]
            norm_data = self.series_transform(series=data, norm_type=norm_type, feature=feature)

        return norm_data

    def series_inverse_transform(self, norm_data, norm_type, feature):
        data = norm_data.copy()
        if norm_type == 'zeta':
            series = data * self.std[feature] + self.mean[feature]
        elif norm_type == 'time':
            series = data * GLOBAL_TIME_MAXIMA[feature]
        elif norm_type == 'none':
            series = data
        else:
            raise Exception(f"Not supported type of normalization '{norm_type}' for feature '{feature}'.")

        return series

    def inverse_transform(self, norm_data: Union[pd.DataFrame, pd.Series], feature=None) -> pd.DataFrame:
        if feature is None:
            data: pd.DataFrame = norm_data.copy()
            for feature in norm_data.columns:
                if feature in self.config:
                    norm_type = self.config[feature]
                    data[feature] = self.series_inverse_transform(norm_data=norm_data[feature],
                                                                  norm_type=norm_type,
                                                                  feature=feature)
        else:
            norm_type = self.config[feature]
            data = self.series_inverse_transform(norm_data=norm_data,
                                                 norm_type=norm_type,
                                                 feature=feature)
        return data
