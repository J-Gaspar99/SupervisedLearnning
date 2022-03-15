#направити алгоритам који идентификује шаку
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
import os, os.path

from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import  precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import seaborn as sns

#%%

def mere_uspesnosti(y_pred, data_test):    
    print('procenat pogodjenih uzoraka: ', accuracy_score(data_test, y_pred))
    print('preciznost mikro: ', precision_score(data_test, y_pred, average='micro'))
    print('preciznost mаkro: ', precision_score(data_test, y_pred, average='macro'))
    print('osetljivost mikro: ', recall_score(data_test, y_pred, average='micro'))
    print('osetljivost makro: ', recall_score(data_test, y_pred, average='macro'))
    print('f mera mikro: ', f1_score(data_test, y_pred, average='micro'))
    print('f mera makro: ', f1_score(data_test, y_pred, average='macro'))
    print('')
    print('----------------------------------------------------------------------------------')

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

#reading all images from folder
nazivi_slika = create_file_list("slike_baza")

nazivi_slika = pd.DataFrame(nazivi_slika, columns=['Nazivi slika'])

#convetion of images into matrix
for i in nazivi_slika['Nazivi slika']:
    #odsecamo 20 kolona piksela sa leve i desne strane, vratiti se na ovo ([:, 20:180])
    print(i)
    v.append(np.reshape(np.asarray(Image.open(i)), -1))
  
x_raw = pd.DataFrame(v)

#%%

image = Image.open('slike_baza\test4_9000000_Hand_0009769.jpg')


#%%
podeljeno = nazivi_slika['Nazivi slika'].str.split('_', expand=True)
podeljeno.drop([0, 3], axis=1, inplace=True)
podeljeno.columns = ['skup', 'klasa', 'id_slike']

y = podeljeno.loc[:, 'klasa']
klase = podeljeno.loc[:, 'klasa'].unique()

groupped = podeljeno.groupby('skup').count()
raspodela_po_klasama = groupped['klasa']

#%%
print(len(x_raw.columns))

#%%
x_area = (~x_raw.isin([255])).sum(axis=1) / len(x_raw.columns)
x_mean = x_raw.mean(axis=1)
x_stdev = x_raw.std(axis=1)
x_median = x_raw.median(axis=1)

#x = pd.DataFrame(data={'mean': x_mean, 'stdev': x_stdev})
# x = pd.DataFrame(data={'mean': x_mean, 'stdev': x_stdev, 'median': x_median, 'mean_stdev': x_mean*x_stdev,
#                        'mean_median': x_mean*x_median, 'median_stdev': x_median*x_stdev})
x = pd.DataFrame(data={'mean': x_mean, 'stdev': x_stdev, 'median': x_median})

#%%
#bez pozadine
without_background = x_raw.replace(255, np.nan)
x_mean = without_background.mean(axis=1)
x_stdev = without_background.std(axis=1)
x_median = without_background.median(axis=1)
#random = pd.Series([240]*1208)

x = pd.DataFrame(data={'mean': x_mean, 'stdev': x_stdev, 'median': x_median, 'mean_stdev': x_mean*x_stdev,
                       'mean_median': x_mean*x_median, 'median_stdev': x_median*x_stdev})

#%%
validacioni_skup_x = x.iloc[:raspodela_po_klasama[0], :]
validacioni_skup_y = y[:raspodela_po_klasama[0]]

trening_skup_x = x.iloc[raspodela_po_klasama[0]:, :]
trening_skup_y = y[raspodela_po_klasama[0]:]

#%%

acc = []
fin_conf_mat = np.zeros((len(klase), len(klase)))
  
x_test = trening_skup_x.iloc[:151, :]
y_test = trening_skup_y.iloc[:151]

x_train_first = []
x_train_second = trening_skup_x.iloc[152:, :]

y_train_first = []
y_train_second = trening_skup_y.iloc[152:]

x_train = x_train_second
y_train = y_train_second

classifier = MLPClassifier(hidden_layer_sizes=(32, 32, 32), activation='tanh',
                          solver='adam', batch_size=50, learning_rate='constant', 
                          learning_rate_init=0.001, max_iter=50, shuffle=True,
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
print('klase:', classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))


#%%
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
    
    classifier = MLPClassifier(hidden_layer_sizes=(100, 250), activation='tanh',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              learning_rate_init=0.001, max_iter=50, shuffle=True,
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
print('klase:', classifier.classes_)
#disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
#fig, ax = plt.subplots(figsize=(10,10))
#disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
#plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))

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

print(osetljivost_po_klasi(fin_conf_mat, klase))


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
print('klase:', classifier.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
# fig, ax = plt.subplots(figsize=(10,10))
# disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
# plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))


#%%
x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=50, test_size=0.2,stratify=y)
x_train1,x_val,y_train1,y_val=train_test_split(x_train, y_train, random_state=50, test_size=0.2,stratify=y_train)

#%%
# Употреба алгоритма К најближих суседа

#тестирање разних модела
error = []
for w in['uniform','distance']:
  for i in range(1, 7):
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

#%%

print(len(y_test.unique()))