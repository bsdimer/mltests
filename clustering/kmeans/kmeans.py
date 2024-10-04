import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('../Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]]

lx = dataset.iloc[:, 3]
ly = dataset.iloc[:, 4]
print(lx)

plt.scatter(lx, ly)
plt.title("Raw plot")
plt.xlabel("column 3")
plt.ylabel("column 4")
plt.show()

# Using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    k = KMeans(n_clusters=i, init="k-means++", random_state=42)
    k.fit(X)
    wcss.append(k.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow method")
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()

# Now w know that the optimal cluster size is 5
km = KMeans(n_clusters=4, init="k-means++", random_state=42)
y_kmeans = km.fit_predict(X)
print(y_kmeans)
print(X)
X = X.values

# Plot the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='orange', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='green', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')

# Add title and legend
plt.title("Clusters of Mall Customers")
plt.legend()

# Show the plot
plt.show()