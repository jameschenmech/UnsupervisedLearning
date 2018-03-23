# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 12:57:03 2018

@author: junch
"""

import pandas as pd
import numpy as np

df = pd.read_csv('wine.csv')

samples =np.array(df[['total_phenols','od280']])

from sklearn.decomposition import PCA
# =============================================================================
# #PCA model decorrelates the the data
# =============================================================================
model = PCA()

model.fit(samples)

transformed = model.transform(samples)

# =============================================================================
# #Correlation
# =============================================================================
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv('seeds-width-vs-length.csv', header=None)

grains = np.array(df)

#assign the 0th column of grains: width
width = grains[:,0]

#assign the 1st column of grains: length
length = grains[:,1]

#Scatter plot width vs length
plt.figure()
plt.scatter(width, length)
plt.axis('equal')
plt.show()

#Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

#display the correlation
print('\nCorrelation: ',correlation)

#Create PCA instance: model
model = PCA()

#apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

#assign 0th column of pca_features: xs
xs = pca_features[:,0]

#assign 1st column of pca_features: ys
ys = pca_features[:,1]

#scatter plot xs vs ys
plt.figure()
plt.scatter(xs,ys)
plt.axis('equal')
plt.show()

#Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs,ys)

#display the correlation
print(correlation)

# =============================================================================
# #Plot variances of PCA features
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# Create a normalizer: normalizer
normalizer = Normalizer()

samples = pd.read_csv('wine.csv', header=0)

#drop the columns that don't have much variance
#samples.drop((samples.columns[[0,1,2,-1]]), axis=1, inplace=True)
samples.drop((samples.columns[[0,1]]), axis=1, inplace=True)
pca = PCA()

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, pca)

# Fit pipeline to the daily price movements
pipeline.fit(samples)

#create a range of pca components
features = range(pca.n_components_)

#bar plot of the features
plt.figure()
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

# =============================================================================
# #Grains - plotting principal axis
# =============================================================================
grains = np.array(pd.read_csv('seeds-width-vs-length.csv'))

#Make a scatter plot of the un transformed points
plt.figure()
_ = plt.scatter(grains[:,0], grains[:,1])

#Create a PCA instance: model
model = PCA()

#fit model to points
model.fit(grains)

# get the mean of the grain samples: mean
mean = model.mean_

#get the first principal component: first_pc
first_pc = model.components_[0,:]

#plot first_pc as an arrow, starting at mean
_ = plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', 
              width=0.01)

#Keep axes on same sclae
_ = plt.axis('equal')
plt.show()

# =============================================================================
# #Fish dataset - variance of PCA features
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd


#import the fish data
samples = pd.read_csv('fish.csv', header=None)
samples.drop(samples.columns[0], axis=1, inplace = True)

#create scaler: scaler
scaler = StandardScaler()

#create a PCA instance: PCA
pca = PCA(n_components=4)

#create a pipeline:  pipeline
pipeline = make_pipeline(scaler, pca)

#fit the pipe to samples
pipeline.fit(samples)

#print shape of pca_features
print(pca_features.shape)

#Plot the explained variances
features = range(pca.n_components_)

plt.figure()
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.title('fish data')
plt.show()

# =============================================================================
# #Dimension reduction of iris dataset
# =============================================================================
# reduce to intrinsic dimensions if known

from sklearn.decomposition import PCA
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data
y = iris.target
#iris.target_names contains what these factors are called in a list

species = iris.target_names

pca = PCA(n_components=2)

pca.fit(samples)

transformed = pca.transform(samples)

print(transformed.shape)

xs = transformed[:,0]
ys = transformed[:,1]

plt.figure()
plt.scatter(xs,ys,c=y)
plt.show()

# =============================================================================
# #TruncatedSVD and csr_matrix
# =============================================================================
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

#create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

#create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

#create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

#wikipedia-vectors.csv contains the tdidf matrix in array form
#need to convert it back to csr/sparse matrix
articles = pd.read_csv('wikipedia-vectors.csv', header=[0],
                                index_col=0)

#get the words in the tfidf matrix
words = articles.columns.get_values().tolist()

#convert articles to np arrary
articles = np.array(np.transpose(articles))

#convert back to sparse matrix
from scipy import sparse

csr_mat = sparse.csr_matrix(articles)

#fit the pipeline to articles
pipeline.fit(csr_mat)

#calculate the cluster labels: labels
labels = pipeline.predict(csr_mat)

#create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label':labels, 'article':words})

#display df sorted by cluster label
print(df.sort_values('label'))

# =============================================================================
# #tf-idf arrary
# =============================================================================
#import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#create a TFidfVectorizer
tfidf =  TfidfVectorizer()

documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']

#Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

#print result of toarray() method
print(csr_mat.toarray())

#Get the words: words
words = tfidf.get_feature_names()

#print words
print(words)

# =============================================================================
# #close all figures after waiting 15 seconds
# =============================================================================
plt.pause(10)
plt.close('all')