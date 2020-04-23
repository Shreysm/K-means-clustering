#Author:Shreyas Mohan

from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import csv
from sklearn.cluster import KMeans#For elbow method only
from scipy.spatial.distance import cdist


# To determine the value of k graphically
def elbow_method():
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

file1=open('iris.data')
In_text = csv.reader(file1,delimiter = ',')
 
file2 =open('iris.csv','w')
out_csv = csv.writer(file2)
out_csv.writerow(['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']) 
file3 = out_csv.writerows(In_text)
 
file1.close()
file2.close()
df = pd.read_csv("iris.csv") #load the dataset
df.head()

# Change categorical data to number 0-2
df["Species"] = pd.Categorical(df["Species"])
df["Species"] = df["Species"].cat.codes
# Change dataframe to numpy matrix
data = df.values[:, 0:4]
category = df.values[:, 4]

elbow_method()

# Number of clusters
k = 3
# Number of training data
n = data.shape[0]
# Number of features in the data
c = data.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
incorrect = 0
for i, j in zip(category,clusters):
    if i != j:
        incorrect +=1
print("Number of data points incorrectly clustered")
print(incorrect)
# Plot the data and the centers generated as random
colors=['red', 'blue', 'green']
for i in range(n):
    if colors[int(category[i])] == 'red':
        l1 = plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])], label = 'Iris-setosa')
    elif colors[int(category[i])] == 'blue':
        l2 = plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])], label = 'Iris-versicolor')
    elif colors[int(category[i])] == 'green':
        l3 = plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])], label = 'Iris-virginica')
handles, labels = plt.gca().get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
plt.legend(handle_list, label_list)
plt.scatter(centers_new[:,0], centers_new[:,1], marker='^', c='black', s=150, label = 'Centroids')
plt.savefig('KMeans')
plt.show()
#To print the centroids
print(centers_new)