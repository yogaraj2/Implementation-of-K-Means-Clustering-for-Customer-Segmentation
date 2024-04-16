# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Pick customer segment quantity (k).
2.Seed cluster centers with random data points.
3.Assign customers to closest centers.
4.Re-center clusters and repeat until stable. 
```

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Yogaraj .S
RegisterNumber:  212223040248
*/

import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X = data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data ['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r','g','b','c','m']
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```
## Output:
![K Means Clustering for Customer Segmentation](sam.png)

![Screenshot 2024-04-16 201038](https://github.com/yogaraj2/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153482637/69d99eb3-b61e-49f1-920f-b557e5d3a7bb)

![Screenshot 2024-04-16 201049](https://github.com/yogaraj2/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153482637/195f550d-fc60-481f-92c2-e0fca41137e4)
 
![Screenshot 2024-04-16 201057](https://github.com/yogaraj2/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153482637/e71b6e92-acbe-4b12-b82d-c1a9f9bdc96b)

![image](https://github.com/yogaraj2/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/153482637/575068fe-a9ce-4069-ac0a-390bc096f999)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
