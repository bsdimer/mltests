import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Example: Dataset
print("\nDataset:")
print(dataset)

# Extract features and Dependent Variable
X = dataset.iloc[:, 1:-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Variables

# Check results
print(X)
print(y)

# Reshape y - [[y]]
y = y.reshape(len(y), 1)
print(y)

# Apply Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Check results
print(X)
print(y)

# Implement SVR predictions
from sklearn.svm import SVR

regression = SVR(kernel='rbf')
regression.fit(X, y)

# Predict single value
scaledPredictValue = sc_X.transform([[6.5]])
unscaledPredict = regression.predict(scaledPredictValue).reshape(-1, 1)
pred = sc_y.inverse_transform(unscaledPredict)
print(pred)

print(regression.predict(X).reshape(-1, 1))
print(sc_y.inverse_transform(regression.predict(X).reshape(-1, 1)))

# Visualize
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regression.predict(X).reshape(-1, 1)), color='green')
plt.title("SVR prediction")
plt.xlabel("Years of experience")
plt.ylabel("salaries")
plt.show()
