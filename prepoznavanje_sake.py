# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 22:59:42 2022

@author: Jovan
"""

#направити алгоритам који идентификује шаку


import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pathlib
import sklearn 
from PIL import Image
import os, os.path, time



def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


v=[]
#reading all images from folder
lista=createFileList("slike_baza")

vrednosti=pd.DataFrame(lista)

#convetion of images into matrix
for i in vrednosti[0]:
    v.append(np.reshape(np.asarray(Image.open(i)),-1))
  
f=pd.DataFrame(v)

#%%
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(100, 70))
plt.title("Persons Dendograms")
dend = shc.dendrogram(shc.linkage(f, method='ward', metric='euclidean'))
plt.show()


#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=151, n_init=10, max_iter=500, 
                init='k-means++',random_state=42)
# init moze biti 'random' uzorci ili odredjene vrednosti date matricom
kmeans.fit_predict(f)
#print(kmeans.cluster_centers_)
print(kmeans.labels_)
sake_osobe=pd.DataFrame(kmeans.labels_)

#%%
plt.plot(range(0,30000),f[0,:])


#plt.figure(figsize=(100, 70))
#plt.scatter(f[:,0], f[:,1], c=cluster.labels_, cmap='rainbow')
#plt.xlabel("Income")
#plt.ylabel("Spending score")
#plt.show()


#plt.figure(figsize=(100,70))
#plt.scatter(f)
#plt.xlabel("")
#plt.ylabel("")
#plt.show()


