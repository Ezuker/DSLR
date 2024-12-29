import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse as args
from utils import load_csv


def histogram(csv_file: pd.DataFrame, feature: str):
	try:
		fig, ax = plt.subplots()
		slytherin = csv_file[csv_file['Hogwarts House'] == 'Slytherin'][feature]
		gryffindor = csv_file[csv_file['Hogwarts House'] == 'Gryffindor'][feature]
		ravenclaw = csv_file[csv_file['Hogwarts House'] == 'Ravenclaw'][feature]
		hufflepuff = csv_file[csv_file['Hogwarts House'] == 'Hufflepuff'][feature]
		max = csv_file[feature].max()
		min = csv_file[feature].min()
		bins = np.arange(min, max, (max - min) / 25)
		ax.hist(slytherin, bins, histtype='step', color='green', label="Slytherin")
		ax.hist(gryffindor, bins, histtype='step', color='red', label="Gryffindor")
		ax.hist(ravenclaw, bins, histtype='step', color='yellow', label="Ravenclaw")
		ax.hist(hufflepuff, bins, histtype='step', color='blue', label="Hufflepuff")
		ax.set_title(f"Histogram comparing {feature}")
		ax.set_xlabel(f"Mark of {feature}")
		ax.set_ylabel(f"Number of student")
		ax.legend()
		plt.show()
	except KeyError as e:
		print(f"Please put a valid feature: {e}")

def main():
	parser = args.ArgumentParser(description="usage: python3 histogram.py --feature \"Astronomy\"")
	parser.add_argument('--file', type=str, help="location of the dataset", required = True)
	parser.add_argument('--feature', type=str, help="name of the feature", required = True)
	arg = parser.parse_args()
	csv_file = load_csv.load(arg.file)
	histogram(csv_file, arg.feature)


if __name__ == '__main__':
	main()