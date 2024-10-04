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

# Create linear regression prediction
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Create Polynomial regression predictions
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
print(X_poly)
print(y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Plot Linear
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
# plt.show()

# Plot Linear
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='green')

# Make curve smoother
X_grid = np.arange(min(X), max(X), 0.1)
print(X)
print(X_grid)
X_grid = X_grid.reshape((len(X_grid), 1))
print(X_grid)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='brown')
plt.show()

# Predicting the concrete values in order to validate the model
print(lin_reg.predict([[6.5]]))

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
