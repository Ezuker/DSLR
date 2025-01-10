import numpy as np
import pandas as pd
from utils import load_csv
from math import exp, log, prod



def model(weight: np.array, x: np.array):
	"""
	Param: 
	 - weight, a vector of weight (belong to a specifiq class)
	 - x, a vector of values of features (belong to a specifiq class)

	Return:
	The probability of belonging to a class
	"""
	return 1 / (1 + exp(-(weight.T.dot(x))))


def getProba(data: pd.DataFrame, weight: np.array):
	features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
						'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
						'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
						'Flying']
	result = []
	for feature in features:
		x = data[feature]
		result.append(model(weight, x))
	return result


def loss(data: pd.DataFrame, weight: np.array):
	return -log(prod(getProba(data, weight)))


def gradient_descent(data: pd.DataFrame, weight: np.array, i: int, alpha: float):
	proba = []
	for _ in range(i):
		weight = weight - alpha * 
	return proba


def main():
	try:
		data_train_file = load_csv.load()
		houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
		weight = np.zeros(12)
		for house in houses:
			proba = gradient_descent(data_train_file[house], weight, 100000, 0.01)
			print(proba)
		
	except Exception as e:
		print(e)


if __name__ == "__main__":
	main()

#Program that takes dataset_train.csv and ouput a weight.csv
# output:
# class 1, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying
# class 2, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying
# class 3, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying
# class 4, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying

# We will have nb_class * nb_features = 4 * 12 = 48 weights