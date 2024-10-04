import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Example: Dataset
print("\nDataset:")
print(dataset)

# Extract features and Dependent variable
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent variables

# Spitting all the data into Training and Test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test result
y_pred = regressor.predict(X_test)

# Plot the results
# Това полага точките върху графиката. т.е. слагаме всички точки от реалните данни
plt.scatter(X_train, y_train, color='red')
# Това чертае графика със предикшъните, т.е. y e предиктнат.
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years")
plt.ylabel("Salaries")
plt.show()

# Plot the results
# Това полага точките върху графиката. т.е. слагаме всички точки от реалните данни
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years")
plt.ylabel("Salaries")
plt.show()