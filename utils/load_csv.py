import pandas as pd


def load(path: str) -> pd.DataFrame:
    """Open a csv file and return a pd.DataFrame class

    Args:
        path (str): path of the csv file

    Returns:
        pd.DataFrame: The CSV file as a pd.DataFrame
    """
    csv = pd.read_csv(path)
    return csv