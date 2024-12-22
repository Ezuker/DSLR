import pandas as pd
import argparse as args
import numpy as np
from utils import load_csv
import math


def describe(file: pd.DataFrame):
	numeric_data = file.select_dtypes(include=["float", "int"])
	describe_data = pd.DataFrame(columns=numeric_data.columns, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'var', 'IQR'])
	stats = {
		'count': {column: sum(1.0 for x in numeric_data[column] if not np.isnan(x)) for column in numeric_data.columns},
		'mean': 0,
		'std': 0,
		'min': 0,
		'25%': 0,
		'50%': 0,
		'75%': 0,
		'max': 0,
		'var': 0,
		'IQR': 0	}

	for stat, values in stats.items():
		describe_data.loc[stat] = values
		
	print(describe_data)


def main():
	try:
		parser = args.ArgumentParser(description="test")
		parser.add_argument('--csv', type=str, help="location of the .csv file", required = True)
		arg = parser.parse_args()
		test_file = load_csv.load(arg.csv)
		print("Our output\n")
		describe(test_file)
		print("\nExpected output\n")
		print(test_file.describe())
		print(type(test_file.describe())) #Comme un csv donc il faut refaire un .csv mais avec count mean etc
	except FileNotFoundError as e:
		print(e)


if __name__ == "__main__":
	main()