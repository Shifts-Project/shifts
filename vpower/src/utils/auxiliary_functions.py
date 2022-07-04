import pandas as pd


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
