import numpy as np
from math import exp
from prepare_data import prepare_data_without_houses, scale_features
import pandas as pd
import json
import argparse as args

def predict(weights: np.array, x: np.array):
    """
    Predict the house of a student based on their features.

    Parameters:
    weights (np.array): The weight matrix for the logistic regression model.
    x (np.array): The feature vector for a student.

    Returns:
    str: The predicted house for the student.
    """
    try: 
        houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
        probas = []

        for weight in weights:
            proba = 1 / (1 + exp(-np.dot(weight, x)))
            probas.append(proba)

        return houses[np.argmax(probas)]
    except Exception as e:
        print(f"Error: {e}")
        return None


def read_settings():
    try:
        with open(".settings.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return {}


def main():
    try:
        parser = args.ArgumentParser()
        parser.add_argument('--file', type=str, help="location of the dataset", required = True)
        arg = parser.parse_args()
        data = pd.read_csv(arg.file)
        weight = pd.read_csv("weight.csv")
        settings = read_settings()
        features = settings.get("features", []) 

        data = scale_features(data, features)
        data = prepare_data_without_houses(data)

        weight_array = np.array(weight[features])
        results = []
        for _, row in data.iterrows():
            x = np.array(row[features])
            result = predict(weight_array, x)
            results.append((_, result))
        results_df = pd.DataFrame(results, columns=['Index', 'Hogwarts House'])
        results_df.to_csv("houses.csv", index=False)
    except Exception as e:
        print(f"Error: {e}")
    
    




if __name__ == "__main__":
	main()