import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Example: Dataset
print("\nDataset:")
print(dataset)

# Extract features and Dependent variable
X = dataset.iloc[:, 1:-1].values  # Features
y = dataset.iloc[:, -1].values   # Dependent variables

# Train Decision Tree Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=0, n_estimators=10)
regressor.fit(X, y)

print(regressor.predict([[6.5]]))

# Visualize
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.title("DT prediction")
plt.xlabel("Years of experiency")
plt.ylabel("salaries")
plt.show()