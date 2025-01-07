import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse as args
import sys
from utils import load_csv

def scatter_plot(csv_file: pd.DataFrame, feature1: str, feature2: str):
    data = csv_file.dropna()
    HogwartsHouse = { 
        'Gryffindor': {
            'dataFrame': data.loc[data['Hogwarts House'] == 'Gryffindor'],
            'color': 'red'
        },
        'Slytherin': {
            'dataFrame': data.loc[data['Hogwarts House'] == 'Slytherin'],
            'color': 'green'
        },
        'Ravenclaw': {
            'dataFrame': data.loc[data['Hogwarts House'] == 'Ravenclaw'],
            'color': 'blue'
        },
        'Hufflepuff': {
            'dataFrame': data.loc[data['Hogwarts House'] == 'Hufflepuff'],
            'color': 'yellow'
        }
    }
    for house, items in HogwartsHouse.items():
        x = items['dataFrame'][feature1]
        y = items['dataFrame'][feature2]
        plt.scatter(x, y, label=house, color=items['color'])
    plt.legend()
    plt.title(f'Scatter plot : {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

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