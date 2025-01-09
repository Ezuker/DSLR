import pandas as pd


def load(path: str) -> pd.DataFrame:
    """Open a csv file and return a pd.DataFrame class

    Args:
        path (str): path of the csv file

    Returns:
        pd.DataFrame: The CSV file as a pd.DataFrame
    """
    try:
        csv = pd.read_csv(path)
        print(f"Loading dataset of dimensions {csv.shape}")
        return csv
    except FileNotFoundError:
        print("Error while opening the file")
        return None
    except pd.errors.ValueLabelTypeMismatch:
        print("Error while opening the file")
        return None
    except pd.errors.DatabaseError:
        print("Error while opening the file")
        return None
    except pd.errors.DataError:
        print("Error while opening the file")
        return None
    except pd.errors.ClosedFileError:
        print("Error while opening the file")
        return None
    except pd.errors.DuplicateLabelError:
        print("Error while opening the file")
        return None
    except pd.errors.IndexingError:
        print("Error while opening the file")
        return None
    except pd.errors.InvalidColumnName:
        print("Error while opening the file")
        return None
    except pd.errors.IndexingError:
        print("Error while opening the file")
        return None
    except pd.errors.UnsortedIndexError:
        print("Error while opening the file")
        return None
    except pd.errors.ParserError:
        print("Error while opening the file")
        return None
    except ValueError:
        print("Error while opening the file")
        return None
    except TypeError:
        print("Error while opening the file")
        return None
    except AttributeError:
        print("Error while opening the file")
        return None
    except PermissionError:
        print("Error while opening the file")
        return None
    except NotADirectoryError:
        print("Error while opening the file")
        return None
