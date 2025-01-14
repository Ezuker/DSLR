from asyncio import sleep
import numpy as np
import pandas as pd
from prepare_data import prepare_data, scale_features
from utils import load_csv
from math import log, prod





# def getProba2(weight: np.array, x: np.array, y: int):
# 	"""P(y | x;w)

# 	Args:
# 		weight (np.array): list of weight
# 		x (np.array): value of feature
# 		y (int): true or false

# 	Returns:
# 		float: Return the probability
# 	"""
# 	return model(weight, x) ** y * (1 - model(weight, x)) ** (1 - y)


# def getProba(data: pd.DataFrame, weight: np.array):
# 	features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
# 						'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
# 						'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
# 						'Flying']
# 	result = []
# 	for feature in features:
# 		x = data[feature]
# 		result.append(model(weight, x))
# 	return result

# def derLoss(data: pd.DataFrame, feature: str, house: str, weight: np.array):
# 	"""y(n) = 1 if house is same as the data else 0
# 	   x(n) = value of feature


# 	Args:
# 		data (pd.DataFrame): data
# 		house (str): target House
# 		weight (np.array): list of weight
# 	"""
# 	lst = []
# 	for _,row in data.iterrows():
# 		yn = 1 if row['Hogwarts House'] == house else 0
# 		xn = row[feature]
# 		lst.append((yn - model(weight, xn)) * xn)
# 	return -sum(lst)

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
	# return (1/m) * np.sum((y_pred - y) * x)
	return (1/m) * np.dot(x.T, (y_pred - y))


def transformHouseToBinary(value, house):
	return 1 if value == house else 0
	
def gradient_descent(data: pd.DataFrame, house: str, iteration: int, alpha: float) -> list:
	"""Take the data and the House in arg and return a list of weight

	Args:
		data (pd.DataFrame): data of the house
		house (str): The house name
		iteration (int): number of iteration
		alpha (float): hyper parameter alpha

	Returns:
		proba (list): Return a list of probabilities  
	"""
	unwantedFeature = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
	weight = np.zeros(14)
	y = np.array(data["Hogwarts House"].apply(lambda value: transformHouseToBinary(value, house)))
	x = np.array(data.drop(columns=unwantedFeature))
	x = np.insert(x, 0, 1, axis=1)
	for _ in range(iteration):
		y_pred = model(weight=weight, x=x)
		weight = weight - alpha * gradient(y, y_pred, x)
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


def accuracy_rate(weight: np.array, data: pd.DataFrame):
	features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
			'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
			'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
			'Flying']
	houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
	test_data = data[features]
	correct_counter = 0
	for i, (_, student) in enumerate(test_data.iterrows()):
		x = np.array(student.values)
		x = np.insert(x, 0, 1)
		prob = []
		for w_house in weight:
			prob.append(model(w_house, x))
		print(prob)
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


def main():
	# try:
		data_train_file = load_csv.load("datasets/dataset_train.csv")
    
    # Supprimer les lignes avec des valeurs manquantes
		data_train_file = prepare_data(data_train_file)

		# Définir les features à mettre à l'échelle
		features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
					'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
					'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
					'Flying']
		
		# Standardiser les features
		data_train_file = scale_features(data_train_file, features)
		houses = ['Slytherin', 'Ravenclaw', 'Hufflepuff', 'Gryffindor']
		weight = []
		for house in houses:
			weight.append(gradient_descent(data_train_file, house, 10000, 0.01))
		save_weights(weight, houses, features)
		accuracy_rate(weight, data_train_file)
		
	# except Exception as e:
	# 	print(e)


if __name__ == "__main__":
	main()

#Program that takes dataset_train.csv and ouput a weight.csv
# output:
# class 1, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying
# class 2, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying
# class 3, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying
# class 4, Arithmancy_weight,Astronomy_weight,Herbology_weight,Defense Against the Dark Arts_weight,Divination_weight,Muggle Studies_weight,Ancient Runes_weight,History of Magic_weight,Transfiguration_weight,Potions_weight,Care of Magical Creatures_weight,Charms_weight,Flying

# We will have nb_class * nb_features = 4 * 12 = 48 weights