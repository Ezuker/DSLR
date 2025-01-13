import numpy as np
from math import exp
from utils import load_csv


def model(weight: np.array, x: np.array):
	"""
	Param: 
	 - weight, a vector of weight (belong to a specifiq class)
	 - x, a vector of values of features (belong to a specifiq class)

	Return:
	The probability of belonging to a class
	"""
	x = np.array(x).reshape(-1, 1)  # Reshape x to (13, 1)
	return 1 / (1 + exp(-(weight.T.dot(x))))


def main():
    weight = load_csv.load("weight.csv")
    data = load_csv.load("datasets/dataset_test.csv")

    houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
                'Flying']

    first = data[data["Index"] == 0]

    x = []
    for feature in features:
        x.append(first[feature].values[0])

    print("Features for first index:", x)

    prob = []
    for house in houses:
        spe_weight = weight[weight["class"] == house].drop(columns=["class"]).values
        print(spe_weight)
        proba = model(spe_weight, x)
        print(f"{house}: {proba}")


if __name__ == "__main__":
	main()