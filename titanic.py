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
# %% Basic Data Analysis
print("Shape frame_train", frame_train.shape)
print("Shape frame_test", frame_test.shape)

# %% Missing Data Analysis
print("NaN count frame_train", frame_train.shape[0] - frame_train.count())
print("NaN count frame_test", frame_test.shape[0] - frame_test.count())

# %% Fill Empty values
frame_train.fillna({"Cabin": "", "Age": -1})