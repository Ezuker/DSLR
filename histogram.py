import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse as args
from utils import load_csv


def histogram_object(csv_file: pd.DataFrame, feature: str):
	_, ax = plt.subplots()

	slytherin = csv_file[csv_file['Hogwarts House'] == 'Slytherin'][feature]
	gryffindor = csv_file[csv_file['Hogwarts House'] == 'Gryffindor'][feature]
	ravenclaw = csv_file[csv_file['Hogwarts House'] == 'Ravenclaw'][feature]
	hufflepuff = csv_file[csv_file['Hogwarts House'] == 'Hufflepuff'][feature]

	nunique = csv_file[feature].nunique()
	bins = np.arange(0, nunique + 1)

	ax.hist(slytherin, bins, histtype='step', color='green', label="Slytherin")
	ax.hist(gryffindor, bins, histtype='step', color='red', label="Gryffindor")
	ax.hist(ravenclaw, bins, histtype='step', color='yellow', label="Ravenclaw")
	ax.hist(hufflepuff, bins, histtype='step', color='blue', label="Hufflepuff")

	ax.set_title(f"Histogram comparing {feature}")
	ax.set_xlabel(f"{feature}")
	ax.set_ylabel(f"Number of students")
	ax.legend()
	plt.show()


def histogram_float(csv_file: pd.DataFrame, feature: str):
	try:
		_, ax = plt.subplots()
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
	except Exception as e:
		print(e)


def main():
	features = np.array(0)
	try:
		parser = args.ArgumentParser(description="usage: python3 histogram.py --feature \"Astronomy\"")
		parser.add_argument('--file', type=str, help="location of the dataset", required = True)
		parser.add_argument('--feature', type=str, help="name of the feature", required = True)
		arg = parser.parse_args()
		csv_file = load_csv.load(arg.file)
		features = np.array([csv_file.columns])
		feature_type = csv_file[arg.feature].dtypes
		if feature_type == object:
			histogram_object(csv_file, arg.feature)
		else:
			histogram_float(csv_file, arg.feature)
	except KeyError as e:
		print(f"Please put a valid feature", end="\n\n")
		if len(features) > 0:
			print(f"Here's possible features for the file you provided:", end="\n\n")
			print(*features[0], sep=', ')
	except Exception as e:
		print(e)
	except KeyboardInterrupt as e:
		print(e)


if __name__ == '__main__':
	main()