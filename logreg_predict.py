import numpy as np
from math import exp


def model(weight: np.array, x: np.array):
	"""
	Param: 
	 - weight, a vector of weight (belong to a specifiq class)
	 - x, a vector of values of features (belong to a specifiq class)

	Return:
	The probability of belonging to a class
	"""
	return 1 / (1 + exp(-(weight.T.dot(x))))


def main():
	pass


if __name__ == "__main__":
	main()