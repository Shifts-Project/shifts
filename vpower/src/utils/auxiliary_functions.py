import pandas as pd
import numpy as np


def load_data_and_set_index(filepath: str, index_column_name: str) -> pd.DataFrame:
    """
    Loads the csv file from the defined filepath and sets the requested column as index
    :param filepath: The filepath for the dataset to be loaded
    :param index_column_name: Column name to be used as the dataframe's index
    :return: pd.DataFrame with datetime index
    """
    data = pd.read_csv(filepath)
    data.set_index(index_column_name, inplace=True, drop=True)
    data.sort_index(inplace=True)

    return data


def data_normalization(data: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """
    Data scaling process:
        1. Standardization scaling for all features except timeSinceDryDock
        2. Zero max scaling of the time feature considering max value=8 years
    :param data: The dataframe to scale
    :param means: Mean values of the features to be scaled
    :param stds: Standard deviation values of the features to be scaled
    :return: Scaled dataset
    """
    features_norm = data.columns.difference(['timeSinceDryDock'])

    data_norm = (data[features_norm] - means[features_norm]) / stds[features_norm]
    data_norm["timeSinceDryDock"] = data["timeSinceDryDock"] / 4324320

    return data_norm


def denorm_prediction(preds_norm, target_mean, target_std):
    """
    Denormalize model predictions. This is the inverse transformation of the data_normalization function
    :param preds_norm: Normalized predictions. An array with size [number of models x number of samples x 2], where 2
    accounts for [mean, variance]
    :param target_mean: Target mean value to be used
    :param target_std: Target standard deviation to be used
    :return: Denormalized predictions
    """
    preds_denorm = preds_norm.copy()
    preds_denorm[:, :, 0] = (preds_denorm[:, :, 0] * target_std) + target_mean
    preds_denorm[:, :, 1] = preds_denorm[:, :, 1] * np.square(target_std)
    return preds_denorm
