# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:02:58 2018

@author: junch
"""

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

iris = datasets.load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)

from sklearn.cluster import KMeans


# =============================================================================
# #define a cluster model
# =============================================================================
model = KMeans(n_clusters=3)

model.fit(X)

labels = model.predict(X)

print(labels)

#generate an array of new samples
new_samples = np.random.uniform(low=0.1, high=9.9, size=(3,4))
#returns cluster label of new samples
new_labels = model.predict(new_samples)

#generate a scatterplot
xs = X[:,0]
ys = X[:,2]
plt.figure()
_ = plt.scatter(xs,ys,c=labels, alpha=0.5)

#Assign the cluster centers: centroids
centroids = model.cluster_centers_

#Assign the columns of centroids:  centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,2]

plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

# =============================================================================
# #Cross tabulation table:  Evaluating a cluster if labels are given
# #iris.target contains the factors 0,1,2
# =============================================================================
#iris.target_names contains what these factors are called in a list
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df_new = pd.DataFrame({'labels':labels,'species':df.species})

print(df_new)

ct = pd.crosstab(df_new['labels'], df_new['species'])

print(ct)

