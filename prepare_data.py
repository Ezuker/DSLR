import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(csv_file: pd.DataFrame):
    """
    Preprocess the data by filling missing values with the mean of the column corresponding to the house of the student.

    Parameters:
    csv_file (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    try:
        if 'Hogwarts House' not in csv_file.columns:
            raise KeyError("The dataset does not contain a 'Hogwarts House' column.")

        filled_data = csv_file.copy()

        for house in filled_data['Hogwarts House'].unique():
            house_data = filled_data[filled_data['Hogwarts House'] == house]
            
            for column in filled_data.columns:
                if column not in ['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']:
                    if house_data[column].isna().any():
                        mean_value = house_data[column].mean()
                        
                        filled_data.loc[
                            (filled_data['Hogwarts House'] == house) & (filled_data[column].isna()), column
                        ] = mean_value

        return filled_data

    except KeyError as e:
        print(f"Error: {e}")
        return None


def prepare_data_without_houses(csv_file: pd.DataFrame):
    """
    Preprocess the data by filling missing values with the mean of the column.

    Parameters:
    csv_file (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    try:
        filled_data = csv_file.copy()

        for column in filled_data.columns:
            if column not in ['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']:
                if filled_data[column].isna().any():
                    mean_value = filled_data[column].mean()
                    
                    filled_data.loc[filled_data[column].isna(), column] = mean_value

        return filled_data

    except KeyError as e:
        print(f"Error: {e}")
        return None


def scale_features(data, features):
    """
    Scale the features in the dataset.

    Args:
        data (pd.DataFrame): Dataset.
        features (list): List of features to scale.

    Returns:
        pd.DataFrame: Scaled dataset.
    """
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])  # Scale the selected features
    return data