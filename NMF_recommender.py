# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:57:03 2018

@author: junch
"""

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

articles = pd.read_csv('wikipedia-vectors.csv', header=[0],
                                index_col=0)
#get the words in the tfidf matrix
titles = articles.columns.get_values().tolist()

#convert articles to np arrary
articles = np.array(np.transpose(articles))

#create an NMF instance: model
model = NMF(n_components=6)

#fit the model to articles
model.fit(articles)

nmf_features = model.transform(articles)

norm_features = normalize(nmf_features)

current_article = norm_features[23,:] #if has index 23

similarities = norm_features.dot(current_article)

print(similarities) #cosine simularities

# =============================================================================
# #dateframes and labels
# =============================================================================
import pandas as pd

norm_features = normalize(nmf_features)

df = pd.DataFrame(norm_features, index=titles)

current_article = df.loc['Cristiano Ronaldo']

similarities = df.dot(current_article)

print(similarities.nlargest())

# =============================================================================
# #Recommender musical artists
# =============================================================================
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

scaler = MaxAbsScaler()

nmf = NMF(n_components=20)

normalizer = Normalizer()

pipeline = make_pipeline(scaler, nmf, normalizer)

from scipy.sparse import coo_matrix

artists = pd.read_csv('scrobbler-small-sample.csv',header=0)
artists = np.array(artists)

data = artists[:,2]
col = artists[:,1]
indice = artists[:,0]
csr_mat = coo_matrix((data,(col,indice)))
print(csr_mat)

artist_names = (pd.read_csv('artists.csv', header=None)[0]).tolist()

norm_features = pipeline.fit_transform(csr_mat)

df = pd.DataFrame(norm_features, index=artist_names)

artist = df.loc['Bruce Springsteen']

similarities = df.dot(artist)

print(similarities.nlargest())