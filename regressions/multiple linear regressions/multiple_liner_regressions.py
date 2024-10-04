import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('50_Startups.csv')

# Example: Dataset
print("\nDataset:")
print(dataset)

# Extract features and Dependent variable
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Dependent variables


# Preprocessing - Transform label columns to binary representation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("Features (X):")
print(X)
print("\nDependent variable (y):")
print(y)

# Spitting all the data into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print("Predicts: \n")
# Compare predictions with test dependant variable set
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
