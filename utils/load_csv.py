import pandas as pd


def load(path: str) -> pd.DataFrame:
    try:
        csv = pd.read_csv(path)
    except FileNotFoundError:
        print("Error while opening the file")
        raise
    print(f"Loading dataset of dimensions {csv.shape}")
    return csv