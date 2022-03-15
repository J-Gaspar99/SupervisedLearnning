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

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

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
    v.append(np.reshape(np.asarray(Image.open(i)), -1))
  
x_raw = pd.DataFrame(v)

#%%
podeljeno = nazivi_slika['Nazivi slika'].str.split('_', expand=True)
podeljeno.drop([0, 3], axis=1, inplace=True)
podeljeno.columns = ['skup', 'klasa', 'id_slike']

y = podeljeno.loc[:, 'klasa']
klase = podeljeno.loc[:, 'klasa'].unique()

groupped = podeljeno.groupby('skup').count()
raspodela_po_klasama = groupped['klasa']

#%%

img = imread('slike_baza\\test0_0000004_Hand_0000678.jpg')
plt.axis("off")
plt.imshow(img)
print(img.shape)

resized_img = resize(img, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img)
print(resized_img.shape)

fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()









