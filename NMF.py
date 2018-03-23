# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:58:23 2018

@author: junch
"""

from sklearn.decomposition import NMF
import pandas as pd
import numpy as np


model = NMF(n_components=2) #must specify n_components for NMF

df = pd.read_csv('wine.csv')
samples =np.array(df[['total_phenols','od280']])

model.fit(samples)

nmf_features = model.transform(samples)

#dimension of components the same as the dimension of the samples
print(model.components_)

#dot product of nmf_features and model.components reconstructs original sample
re_constructed_sample = np.dot(nmf_features,model.components_)

print(re_constructed_sample)

# =============================================================================
# #Wikipedia articles - NMF
# =============================================================================
#wikipedia-vectors.csv contains the tdidf matrix in array form
#need to convert it back to csr/sparse matrix
articles = pd.read_csv('wikipedia-vectors.csv', header=[0],
                                index_col=0)
#get the words in the tfidf matrix
titles = articles.columns.get_values().tolist()

#convert articles to np arrary
articles = np.array(np.transpose(articles))

from sklearn.decomposition import NMF

#create an NMF instance: model
model = NMF(n_components=6)

#fit the model to articles
model.fit(articles)

nmf_features = model.transform(articles)

print(nmf_features)

# =============================================================================
# #NMF features of the Wikipedia articles
# =============================================================================
#create a pandas dataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

#print the row of 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

#print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

# =============================================================================
# #NMF learns topics of documents
# =============================================================================
#create a DataFrame: components_df

with open('wikipedia-vocabulary-utf8.txt') as file:
    words = file.readlines()
    
components_df = pd.DataFrame(model.components_, columns=words)

#Print the sape of the DataFrame
print(components_df.shape)

#Select row3: component
component = components_df.iloc[3,:]

#print the results of nlargest
print(component.nlargest())

# =============================================================================
# #Explore the LED digits dataset
# =============================================================================
samples = pd.read_csv('lcd-digits.csv',header=None)

samples = np.array(samples)

import matplotlib.pyplot as plt

#Select the 0th row: digit
digit = samples[0,:]

#reshape digit to a 13x8 array:  bitmap
bitmap = digit.reshape(13,8)

#print bitmap
print(bitmap)

#Use plt.imshow to display bitmap
plt.figure()
plt.imshow(bitmap, cmap='gray',interpolation='nearest')
plt.colorbar()
plt.show()

def show_as_image(sample):
    bitmap = sample.reshape(13,8)
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
    
from sklearn.decomposition import NMF

model = NMF(n_components=7)

features = model.fit_transform(samples)

#call show_as_image on each component
for component in model.components_:
    show_as_image(component)
    
#assign the 0th row of features: digit_features
digit_features = features[0,:]



#print the digit_features
print(digit_features)

# =============================================================================
# #PCA doesn't learn parts
# =============================================================================
from sklearn.decomposition import PCA

model = PCA(n_components=7)

features = model.fit_transform(samples)

for component in model.components_:
    show_as_image(component)