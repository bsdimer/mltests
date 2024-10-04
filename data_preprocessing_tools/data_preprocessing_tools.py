import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('Data.csv')

# Handle missing values: convert Age and Salary to numeric, coercing errors to NaN
dataset['Age'] = pd.to_numeric(dataset['Age'], errors='coerce')
dataset['Salary'] = pd.to_numeric(dataset['Salary'], errors='coerce')

# Example: Dataset
print("\nDataset:")
print(dataset)

# Extract features and Dependent variable
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Dependent variables

# Preprocessing - Fix missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Preprocessing - Transform label columns to binary representation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Preprocessing encode Dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print("Features (X):")
print(X)
print("\nDependent variable (y):")
print(y)

# Spitting all the data into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nX_train:")
print(X_train)
print("\nX_test:")
print(X_test)
print("\ny_train:")
print(y_train)
print("\ny_test:")
print(y_test)

# Feature scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
print("\nX_train scaled:")
print(X_train)
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("\nX_test scaled")
print(X_test)




