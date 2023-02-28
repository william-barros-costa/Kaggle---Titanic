# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Constants
LOCATION_TRAIN = r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\train.csv"
LOCATION_TEST = r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\test.csv"

# %% Utils


# %% Import data
frame_train  = pd.read_csv(LOCATION_TRAIN)
frame_test = pd.read_csv(LOCATION_TEST)

frame_train.columns, frame_test.columns
# %% Shape
print("Shape frame_train", frame_train.shape)
print("Shape frame_test", frame_test.shape)

# %% Initial Information Train
print("Train")
frame_train.info()

# %% Initial Information Test
print("Test")
frame_test.info()

# %% Initial Description Train
print("Train")
frame_train.describe()

# %% Initial Description Test
print("Test")
frame_test.describe()

# TODO: Divide into numerical and categorical. Histograms for numerical, heatmap for categorical
