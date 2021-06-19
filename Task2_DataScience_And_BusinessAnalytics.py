#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# ## DATA SCIENCE AND BUSINESS ANALYTICS TASK : 2

# ## Prediction using Unsupervised ML: From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually(K-Means Clustering).
# ## Langauge: Python
# ## IDE: Jupyter Notebook
# ## Libraries/Datasets used: Scikit Learn, Pandas, Numpy, Iris Dataset
# 

# ### By SUPARNA SARKAR

# In[35]:


#importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


# **Loading the Iris dataset into the notebook**

# In[27]:


iris = datasets.load_iris()
iris_df = pd.read_excel(r'C:\Users\Suparna\Documents\SuparnaDataset\Iris.xlsx')
print("Successfully imported data into console" )  


# **Viewing the first few rows from the dataset**

# In[28]:


iris_df.head()


# **Finding the optimal number of clusters for K-means and determining the value of k**

# In[36]:


#finding the optimum number of clusters for k-means classification
x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    OMP_NUM_THREADS=1


# **Plotting the graph onto a line graph to observe the pattern**

# In[37]:


#plotting the results onto a line graph,allowing us to observe The elbow
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# "The elbow method" got its name from the elbow pattern forming something like above. The optimal clusters are formed where the elbow occurs. This is when the WCSS(Within Cluster Sum of Squares) doesn't decrease with every iteration significantly. Here we choose the number of clusters as '3'.

# **Creating K-means Classifier**

# In[31]:


# Applying kmeans to the dataset 
# Creating the kmeans classifier

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# **Visualising the cluster data**

# In[32]:


# Visualising the clusters 
# Preferably on the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')


# In[33]:


# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# **Now combining both the above graphs together**

# In[34]:


# Visualising the clusters 
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()

