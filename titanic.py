# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# %% Divide into numerical and categorical.
Y_train = frame_train["Survived"]
X_train = frame_train.loc[:, frame_train.columns != "Survived"]
X_numerical_train = frame_train.select_dtypes(include=[np.number])
X_categorical_train = frame_train.select_dtypes(exclude=[np.number])

X_numerical_train.columns, X_categorical_train.columns, Y_train.name

# %% See numerical distribution
for column in X_numerical_train.columns:
    X_numerical_train[column].hist()
    plt.title(column)
    plt.show()

# %% See correlation between classes
sns.heatmap(X_numerical_train.corr())

