import numpy as np
import matplotlib as plt
import pandas as pd
import argparse as args
from utils import load_csv


def histogram(csv_file: pd.DataFrame, feature: str):
	data = csv_file[feature]
	print(data)

def main():
	parser = args.ArgumentParser(description="usage: python3 histogram.py --feature \"Astronomy\"")
	parser.add_argument('--file', type=str, help="location of the dataset", required = True)
	parser.add_argument('--feature', type=str, help="name of the feature", required = True)
	arg = parser.parse_args()
	csv_file = load_csv.load(arg.file)
	histogram(csv_file, arg.feature)


if __name__ == '__main__':
	main()