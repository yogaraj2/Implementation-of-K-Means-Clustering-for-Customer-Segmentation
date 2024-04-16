# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3.Import KMeans and use for loop to cluster the data.
4.Predict the cluster and plot data graphs.
5.Print the output and end the program. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: YOGARAJ.S
RegisterNumber:  212223040248
*/
```
```
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers_EX8.csv")
data
x=data[['Annual Income (k$)','Spending Score (1-100)']]
x
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending score (1-100)')
plt.show()
k=5
kmeans=KMeans(n_clusters=k)

kmeans.fit(x)
centroids=kmeans.cluster_centers_
Labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(Labels)
colors=['r','g','b','c','m']
for i in range(k):
    cluster_points=x[labels==i]
    plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster (i+1)')
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


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
