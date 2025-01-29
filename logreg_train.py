from asyncio import sleep
import numpy as np
import pandas as pd
from prepare_data import prepare_data, scale_features
from utils import load_csv
from math import log, prod
import random as rd
import matplotlib.pyplot as plt
import argparse as args


def model(weight: np.array, x: np.array):
	"""sig(wTx) or f(x;w) or P(Y = 1 | x;w)

	Param: 
	 - weight, a vector of weight (belong to a specifiq class)
	 - x, a vector of values of features (belong to a specifiq class)

	Return:
	The probability of belonging to a class
	"""
	z = np.dot(x, weight)
	return 1 / (1 + np.exp(-z))


def loss(y: np.array, y_pred: np.array):
	"""Return loss value

	Args:
		y (np.array): real Y
		y_pred (np.array): Y prediction
	"""
	m = len(y)
	return -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def gradient(y: np.array, y_pred: np.array, x: np.array):
	"""Return the partial derivate of loss on w

	Args:
		y (np.array): real y
		y_pred (np.array): Prediction y
	"""
	m = len(y)
	return (1/m) * np.dot(x.T, (y_pred - y))


def transformHouseToBinary(value, house):
	return 1 if value == house else 0


def plt_cost(cost_history):
	"""
	Display the cost vs iterations
	"""
	plt.subplot()
	plt.plot(cost_history)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('Cost vs Iterations')
	# plt.show()


def stochastic_gradient_descent(data: pd.DataFrame, features: list, house: str, iteration: int, alpha: float) -> list:
	"""
    Args:
        data (pd.DataFrame): Training data
        house (str): Target house name
        iteration (int): Number of iterations
        alpha (float): Learning rate

    Returns:
        np.array: Final weights after optimization
    """
	weights = np.zeros(len(features) + 1)
	y = np.array(data["Hogwarts House"].apply(lambda value: transformHouseToBinary(value, house)))
	x = data[features]
	x = np.insert(x, 0, 1, axis=1)
	cost_history = []
	for _ in range(iteration):
		i = rd.randint(0, len(x) - 1)
		xi = x[i:i+1]
		yi = y[i]
		y_pred = model(weights, xi)
		grad = gradient(np.array([yi]), y_pred, xi)
		weights -= alpha * grad
		cost_history.append(loss(np.array([yi]), y_pred))
	plt_cost(cost_history)
	return weights


def minibatch_gradient_descent(data: pd.DataFrame, features: list, house: str, iteration: int, alpha: float) -> list:
	"""
    Args:
        data (pd.DataFrame): Training data
        house (str): Target house name
        iteration (int): Number of iterations
        alpha (float): Learning rate

    Returns:
        np.array: Final weights after optimization
    """
	weights = np.zeros(len(features) + 1)
	y = np.array(data["Hogwarts House"].apply(lambda value: transformHouseToBinary(value, house)))
	x = data[features]
	x = np.insert(x, 0, 1, axis=1)
	cost_history = []
	batch_size = 10
	for _ in range(iteration):
		i = rd.randint(0, len(x) - 10)
		xi = x[i:i+batch_size]
		yi = y[i:i+batch_size]
		y_pred = model(weights, xi)
		grad = gradient(yi, y_pred, xi)
		weights -= alpha * grad
		cost_history.append(loss(yi, y_pred))
	plt_cost(cost_history)
	return weights


def gradient_descent(data: pd.DataFrame, features: list, house: str, iteration: int, alpha: float) -> list:
	"""Take the data and the House in arg and return a list of weight

	Args:
		data (pd.DataFrame): data of the house
		house (str): The house name
		iteration (int): number of iteration
		alpha (float): hyper parameter alpha

	Returns:
		proba (list): Return a list of probabilities  
	"""
	weight = np.zeros(len(features) + 1)
	y = np.array(data["Hogwarts House"].apply(lambda value: transformHouseToBinary(value, house)))
	x = data[features]
	x = np.insert(x, 0, 1, axis=1)
	cost_history = []
	for _ in range(iteration):
		y_pred = model(weight=weight, x=x)
		weight = weight - alpha * gradient(y, y_pred, x)
		cost_history.append(loss(y, y_pred))
	plt_cost(cost_history)
	return weight


def save_weights(weights, classes, features):
	"""
	Save weights for all classes and features to a CSV file.

	Args:
		weights (list): List of weights for each class.
		classes (list): List of class names.
		features (list): List of feature names.
	"""
	# Prepare header
	header = ["class"] + features
	rows = []
	# Create rows for each class
	for class_name, class_weights in zip(classes, weights):
		# Convert class_weights (numpy array) to a list before appending
		rows.append([class_name] + class_weights[1:].tolist())  # Convert to list using .tolist()

	# Save to CSV
	df = pd.DataFrame(rows, columns=header)
	df.to_csv("weight.csv", index=False)


def accuracy_rate(weight: np.array, data: pd.DataFrame, features: list):
	houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
	test_data = data[features]
	correct_counter = 0
	for i, (_, student) in enumerate(test_data.iterrows()):
		x = np.array(student.values)
		x = np.insert(x, 0, 1)
		prob = []
		for w_house in weight:
			prob.append(model(w_house, x))
		for j, p in enumerate(prob):
			if p == max(prob):
				print(f"Student {i} belongs to {houses[j]}")
				# test if the student belongs to the house
				if data['Hogwarts House'][i] == houses[j]:
					print("Correct")
					correct_counter += 1
				else:
					print("Incorrect")
				break
	print(f"Accuracy: {correct_counter / len(data)}")


import chooser

def main():
	try:
		features, algo, dataset, accuracy = chooser.choose()
		data_train_file = load_csv.load(dataset)
		data_train_file = prepare_data(data_train_file)
		data_train_file = scale_features(data_train_file, features)
		houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
		weight = []
		for house in houses:
			if algo == "sgd":
				weight.append(stochastic_gradient_descent(data_train_file, features, house, 5000, 0.01))
			elif algo == "mgd":
				weight.append(minibatch_gradient_descent(data_train_file, features, house, 1000, 0.01))
			else:
				weight.append(gradient_descent(data_train_file, features, house, 10000, 0.01))
		save_weights(weight, houses, features)
		if accuracy:
			accuracy_rate(weight, data_train_file, features)
		print(f'Model trained successfully with {algo}!')
		
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