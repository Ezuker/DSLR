import numpy as np
import pandas as pd
from utils import load_csv


def main():
	try:
		data_train_file = load_csv.load()
	except FileNotFoundError as e:
		print(e)
	except pd.errors.ParserError as e:
		print(e)
	except pd.errors.EmptyDataError as e:
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