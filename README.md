# DSLR: Data Science x Logistic Regression

## Overview
DSLR (Data Science x Logistic Regression) is a project designed to deepen your understanding of Machine Learning by exploring key tools for data analysis and classification. While the term "Data Science" might be a bit of an overstatement in this context, the goal is to provide foundational knowledge in data exploration before applying machine learning techniques.

This project serves as a continuation of linear regression studies, focusing on implementing a logistic regression model for classification tasks. Additionally, we encourage the development of a personal machine learning toolkit to support future projects.

## Learning Objectives
Through this project, you will:
- Learn how to read and process datasets.
- Visualize data using different techniques.
- Select relevant features and clean unnecessary data.
- Train and evaluate a logistic regression model for classification problems.

## Features
- **Data Exploration**: Load and analyze datasets, handling missing values and feature selection.
- **Data Visualization**: Use graphs and plots to understand data distributions and correlations.
- **Logistic Regression Implementation**: Develop a classification model using logistic regression.
- **Performance Evaluation**: Assess model accuracy and effectiveness using metrics such as precision, recall, and F1-score.

## Installation
To get started, clone this repository and make sure you got the dependencies:
```bash
git clone https://github.com/Ezuker/DSLR.git
source /venv/bin/activate
pip install -r requirements.txt

To visualize data:
 - python3 describe.py [file]
 - python3 histogram.py [file] [feature]
 - python3 scatter_plot.py [file] [feature1] [feature2]
 - python3 pair_plot.py [file]

To train the model:
 - python3 logreg_train.py

After the model trained you can try to predict with another dataset:
 - python3 logreg_predict.py [file]
```

## Dependencies
This project requires:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- tkinter

## License
This project is licensed under the MIT License - see the LICENSE file for details.

