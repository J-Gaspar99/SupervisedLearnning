# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:15:32 2022

@author: Jovan
"""

#%%
import math 
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import IPython.display

#import tensorflow as tf
#from tensorflow.python.framework import ops
#import keras
#from keras import backend as K
import os 
import pandas as pd
from PIL import Image
import os, os.path

import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

from sklearn.decomposition import PCA


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



#%%
def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        #print('Za klasu ', klase[i], ' osetljivost je: ', round(osetljivost_i[i], 3))
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg

#%%
def create_file_list(my_dir, format='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(my_dir, topdown=False):
        for name in files:
            if name.endswith(format): 
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list

v = []
v1=[]
f=[]

#reading all images from folder
nazivi_slika = create_file_list("slike_baza")

nazivi_slika = pd.DataFrame(nazivi_slika, columns=['Nazivi slika'])

#convetion of images into matrix
for i in nazivi_slika['Nazivi slika']:
    #odsecamo 20 kolona piksela sa leve i desne strane, vratiti se na ovo
    v.append(np.reshape(np.asarray(Image.open(i))[:, 20:180], -1))
 
    img=imread(i)    
    
    
   
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=False)
    
    v1.append((np.reshape(np.asarray((hog_image))[:, 20:180], -1)))
    f.append(fd)
    
x_raw = pd.DataFrame(v)
x_raw1 = pd.DataFrame(v1)

#%%
podeljeno = nazivi_slika['Nazivi slika'].str.split('_', expand=True)
podeljeno.drop([0, 3], axis=1, inplace=True)
podeljeno.columns = ['skup', 'klasa', 'id_slike']

y = podeljeno.loc[:, 'klasa']
klase = podeljeno.loc[:, 'klasa'].unique()

groupped = podeljeno.groupby('skup').count()
raspodela_po_klasama = groupped['klasa']


#%%

broj_komponeneti = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
strukture = [(100, 200), (128, 128), (64, 64, 64), (128, 128, 128), (32, 64, 128)]
rezultati = []
#broj_komponeneti = [5, 6, 7, 8, 9, 10]

for struktura in strukture:
    for br_c in broj_komponeneti: 
        print("BROJ KOMPONENTI:", br_c)
        #print(br_c)
        pca=PCA(n_components = br_c)
       # x_raw_reduced = np.array(x_raw)
    
        print("Before PCA", x_raw.shape)
    
        x_raw_reduced = pca.fit_transform(x_raw)
    
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
    
        print('---------------------------')
        print("After PCA", x_raw_reduced.shape)
    
        x_raw_reducedDF = pd.DataFrame(x_raw_reduced)
    
    
    
        validacioni_skup_x = x_raw_reducedDF.iloc[:raspodela_po_klasama[0], :]
        validacioni_skup_y = y[:raspodela_po_klasama[0]]
    
        trening_skup_x = x_raw_reducedDF.iloc[raspodela_po_klasama[0]:, :]
        trening_skup_y = y[raspodela_po_klasama[0]:]
    ##
    
    #Neuralna mreza
        acc = []
        fin_conf_mat = np.zeros((len(klase), len(klase)))
    
        for i, r in enumerate(raspodela_po_klasama[1:]):
        
            x_test = trening_skup_x.iloc[i*r:(i+1)*r, :]
            y_test = trening_skup_y.iloc[i*r:(i+1)*r]
        
            x_train_first = trening_skup_x.iloc[:i*r, :]
            x_train_second = trening_skup_x.iloc[(i+1)*r:, :]
        
            y_train_first = trening_skup_y.iloc[:i*r]
            y_train_second = trening_skup_y.iloc[(i+1)*r:]
        
            x_train = pd.concat([x_train_first, x_train_second])
            y_train = pd.concat([y_train_first, y_train_second])
        
        #print(x_test.shape)
        
            classifier = MLPClassifier(hidden_layer_sizes=struktura, activation='tanh',
                                  solver='adam', batch_size=50, learning_rate='constant', 
                                  learning_rate_init=0.001, max_iter=75, shuffle=True,
                                  random_state=42, early_stopping=True, n_iter_no_change=10,
                                  validation_fraction=0.1, verbose=False)
        
            classifier.fit(x_train.values, y_train)
            y_pred = classifier.predict(x_test.values)
            plt.figure
            plt.plot(classifier.validation_scores_)
            #plt.plot(classifier.loss_curve_)
            plt.show()
            print(accuracy_score(y_test, y_pred))
            fin_conf_mat += confusion_matrix(y_test, y_pred)
    
    #print('konacna matrica konfuzije: \n', fin_conf_mat)
    #    print('klase:', classifier.classes_)
    #disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
    #fig, ax = plt.subplots(figsize=(10,10))
    #disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
    #plt.show()
    
        print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
    
    
        acc = []
        fin_conf_mat = np.zeros((len(klase), len(klase)))
        
        classifier = MLPClassifier(hidden_layer_sizes=struktura, activation='tanh',
                                  solver='adam', batch_size=50, learning_rate='constant', 
                                  learning_rate_init=0.001, max_iter=75, shuffle=True,
                                  random_state=42, early_stopping=True, n_iter_no_change=10,
                                  validation_fraction=0.1, verbose=False)
        
        classifier.fit(trening_skup_x.values, trening_skup_y)
        y_pred = classifier.predict(validacioni_skup_x.values)
        plt.figure
        plt.plot(classifier.validation_scores_)
        #plt.plot(classifier.loss_curve_)
        plt.show()
        print(accuracy_score(validacioni_skup_y, y_pred))
        fin_conf_mat += confusion_matrix(validacioni_skup_y, y_pred)
        
        #print('konacna matrica konfuzije: \n', fin_conf_mat)
        #print('klase:', classifier.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
        # fig, ax = plt.subplots(figsize=(10,10))
        # disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
        # plt.show()
        
        tacnost = sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat))
        osetljivost = osetljivost_po_klasi(fin_conf_mat, klase)
        print('procenat tacno predvidjenih: ', tacnost)
        print(osetljivost)
        rezultati.append((br_c, struktura, tacnost, osetljivost))


#%%


#%%
#finalni
acc = []
fin_conf_mat = np.zeros((len(klase), len(klase)))

classifier = MLPClassifier(hidden_layer_sizes=(100, 250), activation='tanh',
                          solver='adam', batch_size=50, learning_rate='constant', 
                          learning_rate_init=0.001, max_iter=50, shuffle=True,
                          random_state=42, early_stopping=True, n_iter_no_change=10,
                          validation_fraction=0.1, verbose=False)

classifier.fit(trening_skup_x.values, trening_skup_y)
y_pred = classifier.predict(validacioni_skup_x.values)
plt.figure
plt.plot(classifier.validation_scores_)
#plt.plot(classifier.loss_curve_)
plt.show()
print(accuracy_score(validacioni_skup_y, y_pred))
fin_conf_mat += confusion_matrix(validacioni_skup_y, y_pred)

#print('konacna matrica konfuzije: \n', fin_conf_mat)
#print('klase:', classifier.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
# fig, ax = plt.subplots(figsize=(10,10))
# disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
# plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))

#%%


#print(osetljivost_po_klasi(fin_conf_mat, klase))











