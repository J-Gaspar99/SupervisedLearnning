# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:14:48 2022

@author: Jovan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import  precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from joblib import dump, load

def mere_uspesnosti(y_pred,data_test):    
 
    
    print('procenat pogodjenih uzoraka: ', accuracy_score(data_test, y_pred))
    print('preciznost mikro: ', precision_score(data_test, y_pred, average='micro'))
    print('preciznost mаkro: ', precision_score(data_test, y_pred, average='macro'))
    print('osetljivost mikro: ', recall_score(data_test, y_pred, average='micro'))
    print('osetljivost makro: ', recall_score(data_test, y_pred, average='macro'))
    print('f mera mikro: ', f1_score(data_test, y_pred, average='micro'))
    print('f mera makro: ', f1_score(data_test, y_pred, average='macro'))
    print('')
    print('----------------------------------------------------------------------------------')




#KNN klasifikator

recepti=pd.read_csv('recipes.csv')

print (recepti['country'].unique())

recall=[]


x=recepti.iloc[:,1:-1].copy()
y=recepti.iloc[:,151].copy()

y[y=='chinese']=0
y[y=='japanese']=1
y[y=='british']=2
y[y=='thai']=3
y[y=='mexican']=4
y[y=='greek']=5
y[y=='french']=6
y[y=='southern_us']=7
y[y=='italian']=8
y=y.astype('int')

x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=50, test_size=0.1,stratify=y)
x_train1,x_val,y_train1,y_val=train_test_split(x_train, y_train, random_state=50, test_size=0.1,stratify=y_train)


#for kuhinja in ['chinese','japanese','thai','british','mexican','greek','italian','southern_us','french']:

#тестирање разних модела
error = []
for w in['uniform','distance']:
  for i in range(1, 10):
      for a in ['ball_tree','brute']:
          for m in ['hamming','euclidean']:
#             
              knn = KNeighborsClassifier(n_neighbors=i,metric=m,weights=w,algorithm=a)
              knn.fit(x_train1, y_train1)
              pred_i = knn.predict(x_val)
              error.append(accuracy_score(y_val, pred_i))
              print('BROJ SUSEDA:'+str(i)+' METRIKA:'+m+' TEZINE:'+w+' AlGORITAM:'+a)
              mere_uspesnosti(pred_i,y_val)
                
plt.figure(figsize=(12, 6))
plt.plot(range(1, 73), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
plt.title('Accurecy score for each model')
plt.xlabel('K Value')
plt.ylabel('Acc')
plt.show()

#обука коначног модела
knnf=KNeighborsClassifier(n_neighbors=7,metric='hamming',weights='distance',algorithm='brute')
knnf.fit(x_train, y_train)


#чување коначног модела
dump(knnf,'KNNKlasifikator.joblib')