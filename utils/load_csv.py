import pandas as pd


def load(path: str) -> pd.DataFrame:
    try:
        csv = pd.read_csv(path)
    except FileNotFoundError:
        print("Error while opening the file")
        raise
    return csv