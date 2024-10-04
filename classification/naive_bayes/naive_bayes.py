import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
dataset = pd.read_csv("../decistion_tree/Social_Network_Ads.csv")

# Split dataset to training set and test set
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=1)
print(X_test)
print(y_test)

# Feature scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predict single result
print(classifier.predict(sc.transform([[30,87000]])))

# Predict all
y_pred = classifier.predict(X_test)

# Compare prediction and test vectors
comparison = np.column_stack(((y_pred.reshape(len(y_pred),1)), y_test))
print(comparison)

# Calculate confusion matrix and accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
print("accuracy_score: ")
print(accuracy_score(y_test,y_pred))
print("confusion_matrix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Training set classification
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

from matplotlib.colors import ListedColormap

# Create mesh grid with scaled values
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# Apply inverse transform to the mesh grid to get original values
X1_orig, X2_orig = sc.inverse_transform(np.c_[X1.ravel(), X2.ravel()])[:, 0].reshape(X1.shape), \
                   sc.inverse_transform(np.c_[X1.ravel(), X2.ravel()])[:, 1].reshape(X2.shape)

# Plot the decision boundary using the original values
plt.contourf(X1_orig, X2_orig, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1_orig.min(), X1_orig.max())
plt.ylim(X2_orig.min(), X2_orig.max())

# Apply inverse transform to the training data to get original values
X_set_orig = sc.inverse_transform(X_set)

# Plot the training data using the original values
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set_orig[y_set == j, 0], X_set_orig[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Plot Test set classification
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# Apply inverse transform to the mesh grid to get original values
X1_orig, X2_orig = sc.inverse_transform(np.c_[X1.ravel(), X2.ravel()])[:, 0].reshape(X1.shape), \
                   sc.inverse_transform(np.c_[X1.ravel(), X2.ravel()])[:, 1].reshape(X2.shape)

plt.contourf(X1_orig, X2_orig, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1_orig.min(), X1_orig.max())
plt.ylim(X2_orig.min(), X2_orig.max())

# Apply inverse transform to the training data to get original values
X_set_orig = sc.inverse_transform(X_set)

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set_orig[y_set == j, 0], X_set_orig[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()