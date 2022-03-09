#направити алгоритам који идентификује шаку
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pathlib
import sklearn 
from PIL import Image
import os, os.path, time
from joblib import Memory
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import  precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve,auc
import seaborn as sns

#%%

def plot_multiclass_roc(y_pred, x_val, y_val, n_classes, figsize=(9, 6)):

    suma=0
    # structures
    fpr = dict()
    tpr = dict()
    tres=dict()
    
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
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

    return suma


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
    #odsecamo 20 kolona piksela sa leve i desne strane, vratiti se na ovo
    v.append(np.reshape(np.asarray(Image.open(i))[:, 20:180], -1))
  
x = pd.DataFrame(v)

#%%
podeljeno = nazivi_slika['Nazivi slika'].str.split('_', expand=True)
podeljeno.drop([0, 3], axis=1, inplace=True)
podeljeno.columns = ['skup', 'klasa', 'id_slike']

y = podeljeno.loc[:, 'klasa']
klase = podeljeno.loc[:, 'klasa'].unique()

groupped = podeljeno.groupby('skup').count()
raspodela_po_klasama = groupped['klasa']

#%%
validacioni_skup_x = x.loc[:raspodela_po_klasama[0]-1, :]
validacioni_skup_y = y[:raspodela_po_klasama[0]]

trening_skup_x = x.loc[raspodela_po_klasama[0]:, :]
trening_skup_y = y[raspodela_po_klasama[0]:]


#%%
#Neuralna mreza
acc = []
fin_conf_mat = np.zeros((len(klase), len(klase)))

for i, r in enumerate(raspodela_po_klasama[1:7]):
    
    x_test = x.loc[(i+1)*r:(i+2)*r-1, :]
    y_test = y.loc[(i+1)*r:(i+2)*r-1]
    
    x_train = x.loc[(i+2)*r-1:, :]
    y_train = y.loc[(i+1)*r:(i+2)*r-1]
    
#     classifier = MLPClassifier(hidden_layer_sizes=(32,32,32), activation='tanh',
#                               solver='adam', batch_size=50, learning_rate='constant', 
#                               learning_rate_init=0.001, max_iter=50, shuffle=True,
#                               random_state=42, early_stopping=True, n_iter_no_change=10,
#                               validation_fraction=0.1, verbose=False)
    
#     classifier.fit(x_train.values, y_train)
#     y_pred = classifier.predict(X_train.iloc[test_index, :].values)
#     plt.figure
#     plt.plot(classifier.validation_scores_)
#     #plt.plot(classifier.loss_curve_)
#     plt.show()
#     print(accuracy_score(y_train.iloc[test_index], y_pred))
#     fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)

# #print('konacna matrica konfuzije: \n', fin_conf_mat)
# print('klase:', classifier.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat,  display_labels=classifier.classes_)
# fig, ax = plt.subplots(figsize=(10,10))
# disp.plot(cmap="Blues", values_format='', xticks_rotation='vertical', ax=ax)
# plt.show()

# print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))


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
# Употреба алгоритма Логистичке регресије

n_classes = 151

matrice_konfuzije=[]

sume=[]
#тестирање разних модела
for num in(500,1000):
    for solv in ('sag','newton-cg','lbfgs'):
        #for way in ('multinomial'):
            classifier=LogisticRegression(max_iter=num,solver=solv,multi_class='auto')
            #classifier= OneVsRestClassifier(svm.SVC(kernel='linear',probability=True))
            classifier.fit(x_train1,y_train1)          
            y_predicted=classifier.predict(x_test)
          #  y_prob = classifier.predict_proba(x_val)
            sume.append(plot_multiclass_roc(y_predicted, x_test, y_test, n_classes=151, figsize=(9, 6)))   
            conf_mat=confusion_matrix(y_test,y_predicted)
            print('PODESAVANJEM SLEDECIH PARAMETARA:\n MAX_BROJ ITERACIJA:{}  '.format(num))
            print('ALGORITAM ZA RESAVANJE OPTIMIZACIONOG PROBLEMA:{}  '.format(solv))
            print('PRINCP RESAVANJA:{}  '.format('auto'))
            print('dobijeni su sledeci rezultati: \n')
            mere_uspesnosti(y_predicted,y_test)
            matrice_konfuzije.append(conf_mat)





#%%
print(3/8)



