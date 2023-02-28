# %% imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# %% Get data
train_data = pd.read_csv(r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\train.csv")
test_data = pd.read_csv(r"C:\Users\William\Desktop\repository\kaggle_competitions\titanic\resources\test.csv")

# %% Train Model
features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train_data[features])
Y_train = train_data["Survived"]
X_test = pd.get_dummies(test_data[features])

forest_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
forest_model.fit(X_train, Y_train)

# %% Predict
Y_test_predictions = forest_model.predict(X_test)

out_frame = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_test_predictions})
out_frame.to_csv("submission.csv", index=False)