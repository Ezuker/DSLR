import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    try:
        if len(sys.argv) != 4:
            print("Usage: python scatter_plot.py <dataset.csv> <feature1> <feature2>")
            sys.exit(1)
        dataset = pd.read_csv(sys.argv[1])
        feature1 = sys.argv[2]
        feature2 = sys.argv[3]
        if feature1 not in dataset.columns:
            print(f"{feature1} not in dataset")
            sys.exit(1)
        if feature2 not in dataset.columns:
            print(f"{feature2} not in dataset")
            sys.exit(1)
        plt.scatter(dataset[feature1], dataset[feature2])
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"{feature1} vs {feature2}")
        plt.show()
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
	main()