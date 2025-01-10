import pandas as pd
import argparse as args
import numpy as np
from utils import load_csv
from math import sqrt
from functools import reduce


def count_non_nan(data, column):
	return sum(1.0 for x in data[column] if not np.isnan(x))


def calc_mean(data, column):
	count = count_non_nan(data, column)
	if count == 0:
		return 0
	return (1 / count) * sum(x for x in data[column] if not np.isnan(x))


def calc_var(data, column):
	count = count_non_nan(data, column)
	mean = calc_mean(data, column)
	if count == 0:
		return 0
	return sum([(x - mean) ** 2 for x in data[column] if not np.isnan(x)]) / (count - 1)


def calc_std(data, column):
	return sqrt(calc_var(data, column))


def calc_min(data, column):
	non_nan = [x for x in data[column] if not np.isnan(x)]
	if len(non_nan) == 0:
		return 0
	return reduce(lambda x, y: x if x <= y else y, non_nan)


def calc_max(data, column):
	non_nan = [x for x in data[column] if not np.isnan(x)]
	if len(non_nan) == 0:
		return 0
	return reduce(lambda x, y: x if x >= y else y, non_nan)


def calc_quartile(data, column, percent):
	"""
	Return the quartile result along the given percent.

	Args:
		data (pd.DataFrame): The DataFrame containing the data.
		column (str): The name of the column to analyze.
		percent (float): The percentile (between 0 and 1, e.g., 0.25 for 25%).

	Returns:
		float: The calculated percentile value, or None if the column is empty or invalid.
	"""
	non_nan = [x for x in data[column] if not pd.isnull(x)]
	if not non_nan:
		return 0
	sorted_list = sorted(non_nan)
	index = percent * (len(sorted_list) - 1)
	lower = int(index)
	upper = lower + 1
	if upper < len(sorted_list):
		# Linear interpolation :/
		result = sorted_list[lower] + (index - lower) * (sorted_list[upper] - sorted_list[lower])
	else:
		result = sorted_list[lower]
	return result


def calc_IQR(data, column):
	return calc_quartile(data, column, 0.75) - calc_quartile(data, column, 0.25)


def describe(file: pd.DataFrame):
	numeric_data = file.select_dtypes(include=["float", "int"])
	describe_data = pd.DataFrame(columns=numeric_data.columns, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'var', 'IQR'])
	stats = {
		'count': {column: count_non_nan(numeric_data, column) for column in numeric_data.columns},
		'mean': {column: calc_mean(numeric_data, column) for column in numeric_data.columns},
		'std': {column: calc_std(numeric_data, column) for column in numeric_data.columns},
		'min': {column: calc_min(numeric_data, column) for column in numeric_data.columns},
		'25%': {column: calc_quartile(numeric_data, column, .25) for column in numeric_data.columns},
		'50%': {column: calc_quartile(numeric_data, column, .5) for column in numeric_data.columns},
		'75%': {column: calc_quartile(numeric_data, column, .75) for column in numeric_data.columns},
		'max': {column: calc_max(numeric_data, column) for column in numeric_data.columns},
		'var': {column: calc_var(numeric_data, column) for column in numeric_data.columns},
		'IQR': 	{column: calc_IQR(numeric_data, column) for column in numeric_data.columns}}
	for stat, values in stats.items():
		describe_data.loc[stat] = values
	print(describe_data)


def main():
	try:
		parser = args.ArgumentParser(description="usage: python3 describe.py --file ./datasets/dataset_train.csv")
		parser.add_argument('--file', type=str, help="location of the .csv file", required = True)
		arg = parser.parse_args()
		test_file = load_csv.load(arg.file)
		describe(test_file)
	except Exception as e:
		print(e)


if __name__ == "__main__":
	main()