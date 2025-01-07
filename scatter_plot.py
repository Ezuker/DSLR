import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse as args
import sys
from utils import load_csv

def scatter_plot(csv_file: pd.DataFrame, feature1: str, feature2: str):
    """
    Generates a scatter plot for two features from a given DataFrame, colored by Hogwarts House.

    Parameters:
    csv_file (pd.DataFrame): The input DataFrame containing the data.
    feature1 (str): The name of the first feature to plot on the x-axis.
    feature2 (str): The name of the second feature to plot on the y-axis.

    Returns:
    None

    Raises:
    KeyError: If the specified features are not found in the DataFrame.

    The scatter plot will display points colored by Hogwarts House:
    - Gryffindor: red
    - Slytherin: green
    - Ravenclaw: blue
    - Hufflepuff: yellow
    """
    try:
        data = csv_file.dropna()
        HogwartsHouse = {
            'Gryffindor': (data.loc[data['Hogwarts House'] == 'Gryffindor'], 'red'),
            'Slytherin': (data.loc[data['Hogwarts House'] == 'Slytherin'], 'green'),
            'Ravenclaw': (data.loc[data['Hogwarts House'] == 'Ravenclaw'], 'blue'),
            'Hufflepuff': (data.loc[data['Hogwarts House'] == 'Hufflepuff'], 'yellow')
        }
        for house, (dataFrame, color) in HogwartsHouse.items():
            plt.scatter(dataFrame[feature1], dataFrame[feature2], label=house, color=color)
        plt.legend()
        plt.title(f'Scatter plot : {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    except KeyError as e:
        print(f"Please put a valid feature: {e}")

def main():
    try:
        parser = args.ArgumentParser(description="usage: python3 scatter_plot.py --feature1 \"Astronomy\" --feature2 \"Arithmancy\"")
        parser.add_argument('--file', type=str, help="location of the dataset", required = True)
        parser.add_argument('--feature1', type=str, help="name of the feature", required = True)
        parser.add_argument('--feature2', type=str, help="name of the feature", required = True)
        arg = parser.parse_args()
        csv_file = load_csv.load(arg.file)
        scatter_plot(csv_file, arg.feature1, arg.feature2)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
	main()