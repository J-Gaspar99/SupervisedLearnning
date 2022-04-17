# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:49:14 2022

@author: Jovan
"""
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from metrics import tacnost_po_klasi, osetljivost_po_klasi_makro, specificnost_po_klasi, preciznost_po_klasi_makro, f_mera, osetljivost_po_klasi_mikro, preciznost_po_klasi_mikro
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler




class Reduction():
    
        def lda_reduction(X_train, X_test, Y_train, lda_componenets):
            s = StandardScaler()
            s.fit(X_train)
            x_train_std = s.transform(X_train)
            x_test_std = s.transform(X_test)
            
            lda = LDA(n_components = lda_componenets)
            lda.fit(X_train,Y_train)
            
            X_train_r = lda.transform(x_train_std)
            X_test_r = lda.transform(x_test_std)
            
            return [X_train_r, X_test_r]
        
        
        
        def pca_reduction(X, pca_components):
            s = StandardScaler()
            s.fit(X)
            x_std = s.transform(X)
            #print("Before PCA", X.shape)
            pca = PCA(n_components=pca_components)
            X_reduced = pca.fit_transform(x_std)
            #print("After PCA", X_reduced.shape)
            X_reduced = pd.DataFrame(X_reduced)
            return X_reduced