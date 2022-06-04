# Modified from source https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-fairness-aml

from sklearn.ensemble import GradientBoostingRegressor
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

from azureml.core import Workspace, Dataset

ws = Workspace.from_config()

dataset = Dataset.get_by_name(ws, name='Titanic_dataset_filtered')
data = dataset.to_pandas_dataframe()

def clean_data(data):
    x_df = data.dropna()
    y_df = x_df.pop('Survived')
    return x_df,y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=3, help="Maximum depth of each tree that limits number of nodes")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Factor by which each tree's contribution shrinks")

    args = parser.parse_args()

    x, y = clean_data(data)
    print("Separated data into x_df and y_df")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    print("Performed train-test split")

    model = GradientBoostingRegressor(max_depth=args.max_depth, learning_rate=args.learning_rate),

    # Train the model on the data    
    model = model.fit(x_train, y_train)
    print("Model was trained by train data")
    
    accuracy = model.score(x_test, y_test)
    print("Accuracy for test data:", accuracy)

    run = Run.get_context()
    run.log("Accuracy", np.float(accuracy))
    run.log("Max depth:", np.float(args.max_depth))
    run.log("Learning rate:", np.int(args.learning_rate))

if __name__ == '__main__':
    main()
