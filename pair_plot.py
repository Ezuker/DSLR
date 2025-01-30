import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse as args


def histogram(feature: str, ax, csv_file: pd.DataFrame):
    """
    Plots histograms for a specified feature from a DataFrame, separated by Hogwarts House.

    Parameters:
    feature (str): The feature/column name to plot the histogram for.
    ax: The matplotlib axes object where the histogram will be plotted.
    csv_file (pd.DataFrame): The DataFrame containing the data, including the 'Hogwarts House' column.

    Raises:
    KeyError: If the specified feature is not found in the DataFrame.
    """
    try:
        slytherin = csv_file[csv_file['Hogwarts House'] == 'Slytherin'][feature]
        gryffindor = csv_file[csv_file['Hogwarts House'] == 'Gryffindor'][feature]
        ravenclaw = csv_file[csv_file['Hogwarts House'] == 'Ravenclaw'][feature]
        hufflepuff = csv_file[csv_file['Hogwarts House'] == 'Hufflepuff'][feature]
        max = csv_file[feature].max()
        min = csv_file[feature].min()
        bins = np.arange(min, max, (max - min) / 25)
        ax.hist(slytherin, bins, histtype='bar', alpha=0.5 ,color='green', label="Slytherin")
        ax.hist(gryffindor, bins, histtype='bar', alpha=0.5 , color='red', label="Gryffindor")
        ax.hist(ravenclaw, bins, histtype='bar', alpha=0.5 , color='yellow', label="Ravenclaw")
        ax.hist(hufflepuff, bins, histtype='bar', alpha=0.5 , color='blue', label="Hufflepuff")
    except KeyError as e:
        print(f"Please put a valid feature: {e}")


def scatter_plot(feature1: str, feature2: str, ax, csv_file: pd.DataFrame, size: int):
    """
    Generates a scatter plot for two features from a given DataFrame, colored by Hogwarts House.

    Parameters:
    csv_file (pd.DataFrame): The input DataFrame containing the data.
    feature1 (str): The name of the first feature to plot on the x-axis.
    feature2 (str): The name of the second feature to plot on the y-axis.

    Returns:
    None

    Raises:
    KeyError: If the specified features are not found in the DataFrame.

    The scatter plot will display points colored by Hogwarts House:
    - Gryffindor: red
    - Slytherin: green
    - Ravenclaw: blue
    - Hufflepuff: yellow
    """
    try:
        data = csv_file.dropna()
        HogwartsHouse = {
            'Gryffindor': (data.loc[data['Hogwarts House'] == 'Gryffindor'], 'red'),
            'Slytherin': (data.loc[data['Hogwarts House'] == 'Slytherin'], 'green'),
            'Ravenclaw': (data.loc[data['Hogwarts House'] == 'Ravenclaw'], 'blue'),
            'Hufflepuff': (data.loc[data['Hogwarts House'] == 'Hufflepuff'], 'yellow')
        }
        for house, (dataFrame, color) in HogwartsHouse.items():
            ax.scatter(dataFrame[feature1], dataFrame[feature2], label=house, color=color, s=size)
    except KeyError as e:
        print(f"Please put a valid feature: {e}")

def onclick(event, axs, features, data):
    """
    Event handler for mouse click events on a grid of subplots.

    Parameters:
    event (matplotlib.backend_bases.Event): The mouse click event.
    axs (list of list of matplotlib.axes.Axes): 2D list of subplot axes.
    features (list of str): List of feature names corresponding to the data columns.
    data (pandas.DataFrame): The dataset containing the features.

    Behavior:
    - Identifies which subplot was clicked.
    - If the clicked subplot is not on the diagonal, it creates a scatter plot of the corresponding features.
    - If the clicked subplot is on the diagonal, it creates a histogram of the corresponding feature.
    - Displays the new plot in a separate figure window.
    """
    if event.inaxes:
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                if event.inaxes == ax:
                    fig, new_ax = plt.subplots()
                    if i != j:
                        plt.title(f'{features[i]} vs {features[j]}')
                        plt.ylabel(features[i])
                        plt.xlabel(features[j])
                        scatter_plot(features[i], features[j], new_ax, data, 20)
                    else:
                        plt.title(f'{features[i]}')
                        histogram(features[i], new_ax, data)
                    plt.show()
                    break

def pair_plot(csv_file: pd.DataFrame):
    """
    Generates a pair plot for the given DataFrame.
    This function creates a pair plot for a set of predefined features from the input DataFrame.
    It plots histograms on the diagonal and scatter plots on the off-diagonal subplots. The plot
    is displayed using matplotlib.
    Parameters:
    csv_file (pd.DataFrame): The input DataFrame containing the data to be plotted.
    Features:
    - 'Arithmancy'
    - 'Astronomy'
    - 'Herbology'
    - 'Defense Against the Dark Arts'
    - 'Divination'
    - 'Muggle Studies'
    - 'Ancient Runes'
    - 'History of Magic'
    - 'Transfiguration'
    - 'Potions'
    - 'Care of Magical Creatures'
    - 'Charms'
    - 'Flying'
    The function handles missing values by dropping rows with NaN values.
    Raises:
    KeyError: If any of the predefined features are not found in the DataFrame.
    Note:
    - The function connects a click event to the plot for additional interactivity.
    - The legend is placed at the center right of the figure.
    """
    try:
        data = csv_file.dropna()
        features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                    'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
                    'Flying']
        fig, axs = plt.subplots(len(features), len(features), figsize=(20, 20))
        # add some space between the subplots
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i in range(len(features)):
            for j in range(len(features)):
                if i == j:
                    histogram(features[i], axs[i, j], data)
                else:
                    scatter_plot(features[i], features[j], axs[i, j], data, 2)
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                if i == len(features) - 1:
                    axs[i, j].set_xlabel(features[j].replace(" ", '\n'), fontsize=8, labelpad=10)
                    
                if j == 0:
                    axs[i, j].set_ylabel(features[i].replace(" ", "\n"), fontsize=8, labelpad=10)

        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', fontsize=10)
        fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, axs, features, data))
        plt.show()
    except KeyError as e:
        print(f"Please put a valid feature: {e}")


def main():
    try:
        parser = args.ArgumentParser()
        parser.add_argument('--file', type=str, help="location of the dataset", required = True)
        arg = parser.parse_args()
        csv_file = pd.read_csv(arg.file)
        pair_plot(csv_file)
    except Exception as e:
        print(e)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
	main()