# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Constants
LOCATION_GENDER_SUBMISSION = r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\gender_submission.csv"
LOCATION_TRAIN = r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\train.csv"
LOCATION_TEST = r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\test.csv"

# %% Utils


# %% Import data
frame_gender_submission = pd.read_csv(LOCATION_GENDER_SUBMISSION)
frame_train  = pd.read_csv(LOCATION_TRAIN)
frame_test = pd.read_csv(LOCATION_TEST)

frame_gender_submission.columns, frame_train.columns, frame_test.columns
# %% Basic Data Analysis
print("Shape frame_gender_submission", frame_gender_submission.shape)
print("Shape frame_train", frame_train.shape)
print("Shape frame_test", frame_test.shape)

# %%