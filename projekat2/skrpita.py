# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:24:04 2022

@author: Jovan
"""
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from joblib import dump, load
from sklearn.metrics import  precision_score,accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



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




ulaz=input('unesite naziv CSV fajla:')


podaci=pd.read_csv(ulaz+'.csv')

x=podaci.iloc[:,1:-1].copy()
y=podaci.iloc[:,151].copy()

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



for cmodel in('KlasifikatorLogistickeRegresije.joblib','KNNKlasifikator.joblib'):
    klasifikator=load(cmodel)
    if(cmodel=='KlasifikatorLogistickeRegresije.joblib'):
        print('MERE USPESNOSTI KLASIFIKATORA LOGISTICKE REGRESIJE:')
        y_pred=klasifikator.predict(x_test)
        mere_uspesnosti(y_pred,y_test)
        
    if(cmodel=='KNNKlasifikator.joblib'):
        print('MERE USPESNOSTI KLASIFIKATORA K NAJBLIZIH SUSEDA:')
        pred_i = klasifikator.predict(x_val)
        mere_uspesnosti(pred_i,y_val)
        
        
        
        