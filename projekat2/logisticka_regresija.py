# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:45:10 2022

@author: Јован Гашпар IN60/2018 
"""
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score,roc_curve,precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn import svm,datasets
import seaborn as sns
from sklearn.metrics import  precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from joblib import dump, load



def mere_uspesnosti(y_pred,data_test):    
 
    
    print('procenat pogodjenih uzoraka: ', accuracy_score(data_test, y_pred))
    print('preciznost mikro: ', precision_score(data_test, y_pred, average='micro'))
    print('preciznost makro: ', precision_score(data_test, y_pred, average='macro'))
    print('osetljivost mikro: ', recall_score(data_test, y_pred, average='micro'))
    print('osetljivost makro: ', recall_score(data_test, y_pred, average='macro'))
    print('f mera mikro: ', f1_score(data_test, y_pred, average='micro'))
    print('f mera makro: ', f1_score(data_test, y_pred, average='macro'))
    print('')
    print('------------------------------------------------------------------------------------------')

def evaluiraj_ishod(y_test,y_pred):
    conf_mat=confusion_matrix(y_test,y_pred)
   # disp=ConfusionMatrixDisplay.from_predictions(y_test,y_pred,labels=classifier.classes_)
  #  plt.show()
    mere_uspesnosti(conf_mat)
    return conf_mat


def plot_multiclass_roc(y_pred, x_val, y_val, n_classes, figsize=(9, 6)):

    suma=0
    # structures
    fpr = dict()
    tpr = dict()
    tres=dict()
    names=dict()
    names[0]='chinese'
    names[1]='japanese'
    names[2]='british'
    names[3]='thai'
    names[4]='mexican'
    names[5]='greek'
    names[6]='french'
    names[7]='southern_us'
    names[8]='italian'
    
    roc_auc = dict()
    
    y_pred_tmp=y_pred.copy()
    y_val_tmp=y_val.copy()
    
    
    for i in range(n_classes):
        y_pred_tmp=y_pred.copy()
        y_val_tmp=y_val.copy()
        y_pred_tmp[ y_pred!=i]=0
        y_pred_tmp[y_pred==i]=1
        
        y_val_tmp[ y_val!=i]=0
        y_val_tmp[y_val==i]=1
        
        fpr[i], tpr[i], tres[i] = roc_curve(y_val_tmp, y_pred_tmp)
        roc_auc[i] = auc(fpr[i], tpr[i])
        suma=suma+roc_auc[i]

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], names[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

    return suma
   
            
################################################################################################################################



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

x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=10, test_size=0.1,stratify=y)
x_train1,x_val,y_train1,y_val=train_test_split(x_train, y_train, random_state=10, test_size=0.1,stratify=y_train)

n_classes = 9

matrice_konfuzije=[]

sume=[]
#тестирање разних модела
for num in(100,200,500,1000):
    for solv in ('saga','sag','newton-cg','lbfgs'):
        for way in ('ovr','multinomial'):
            classifier=LogisticRegression(max_iter=num,solver=solv,multi_class=way)
            #classifier= OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
            classifier.fit(x_train1,y_train1)          
            y_predicted=classifier.predict(x_test)
          #  y_prob = classifier.predict_proba(x_val)
            sume.append(plot_multiclass_roc(y_predicted, x_test, y_test, n_classes=9, figsize=(9, 6)))   
            conf_mat=confusion_matrix(y_test,y_predicted)
            print('PODESAVANJEM SLEDECIH PARAMETARA:\n MAX_BROJ ITERACIJA:{}  '.format(num))
            print('ALGORITAM ZA RESAVANJE OPTIMIZACIONOG PROBLEMA:{}  '.format(solv))
            print('PRINCP RESAVANJA:{}  '.format(way))
            print('dobijeni su sledeci rezultati: \n')
            mere_uspesnosti(y_predicted,y_test)
            matrice_konfuzije.append(conf_mat)
        
          
#обука коначног модела
klasifikator=LogisticRegression(max_iter=1000,solver='lbfgs',multi_class='multinomial')
klasifikator.fit(x_train,y_train)
#sume.append(plot_multiclass_roc(y_predicted, x_test, y_test, n_classes=9, figsize=(9, 6)))   
#conf_mat=confusion_matrix(y_test,y_predicted)
#print('KONACNI MODEL:\n MAX_BROJ ITERACIJA:{}  '.format(num))
#print('ALGORITAM ZA RESAVANJE OPTIMIZACIONOG PROBLEMA:{}  '.format(solv))
#print('PRINCP RESAVANJA:{}  '.format(way))
#print('dobijeni su sledeci rezultati: \n')
#mere_uspesnosti(y_predicted,y_test)
#matrice_konfuzije.append(conf_mat)


#чување коначног модела
dump(klasifikator,'KlasifikatorLogistickeRegresije.joblib')
                












