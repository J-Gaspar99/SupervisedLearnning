from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from metrics import tacnost_po_klasi, osetljivost_po_klasi_makro, specificnost_po_klasi, preciznost_po_klasi_makro, f_mera, osetljivost_po_klasi_mikro, preciznost_po_klasi_mikro
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import Reduction as red


class Classifier():
              
    
    def start(self, x_reduced, Y, classes, class_distribution):
      #  error_rate = []
        
        x_train_whole = x_reduced.iloc[class_distribution[0]:, :]
        y_train_whole = Y[class_distribution[0]:]
        
        fin_conf_mat = np.zeros((len(classes), len(classes)))
    
        for i, r in enumerate(class_distribution[1:]):
        
            x_validation = x_train_whole.iloc[i*r:(i+1)*r, :]
            y_validation = y_train_whole.iloc[i*r:(i+1)*r]
        
            x_train_first = x_train_whole.iloc[:i*r, :]
            x_train_second = x_train_whole.iloc[(i+1)*r:, :]
        
            y_train_first = y_train_whole.iloc[:i*r]
            y_train_second = y_train_whole.iloc[(i+1)*r:]
        
            x_train = pd.concat([x_train_first, x_train_second])
            y_train = pd.concat([y_train_first, y_train_second])
        
            classifier = self.concrete_classifier()
        
            classifier.fit(x_train.values, y_train)
            y_pred = classifier.predict(x_validation.values)
            
            fin_conf_mat += confusion_matrix(y_validation, y_pred)
            
         #   error_rate.append(np.mean(y_pred != y_validation))
        
        #error_rate = sum(error_rate)/np.size(error_rate)
            
        
            
            
        return fin_conf_mat#, error_rate
    
    def final_model(self, x_reduced, Y, classes, class_distribution):
        #x_reduced = self.pca_reduction(X)
        x_test = x_reduced.iloc[:class_distribution[0], :]
        y_test = Y[:class_distribution[0]]
        x_train_whole = x_reduced.iloc[class_distribution[0]:, :]
        y_train_whole = Y[class_distribution[0]:]
        fin_conf_mat = np.zeros((len(classes), len(classes)))
        classifier = self.concrete_classifier()
        classifier.fit(x_train_whole.values, y_train_whole)
        y_pred = classifier.predict(x_test.values)
        fin_conf_mat += confusion_matrix(y_test, y_pred)
            
        return fin_conf_mat

    def calculate_metrics(self, fin_conf_mat, classes):
        tacnost = tacnost_po_klasi(fin_conf_mat, classes)
        makro_osetljivost = osetljivost_po_klasi_makro(fin_conf_mat, classes)
        mikro_osetljivost = osetljivost_po_klasi_mikro(fin_conf_mat, classes)
        makro_preciznost = preciznost_po_klasi_makro(fin_conf_mat, classes)
        mikro_preciznost = preciznost_po_klasi_mikro(fin_conf_mat, classes)
        specificnost = specificnost_po_klasi(fin_conf_mat, classes)
        makro_f_mera = f_mera(makro_preciznost, makro_osetljivost)
        mikro_f_mera = f_mera(mikro_preciznost, mikro_osetljivost)
        return {'Tacnost': tacnost, 'Makro osetljivost': makro_osetljivost, 'Mikro osetljivost': mikro_osetljivost, 'Makro preciznost': makro_preciznost, 'Mikro preciznost': mikro_preciznost, 'Specificnost': specificnost, 'Makro F-mera': makro_f_mera, 'Mikro F-mera': mikro_f_mera}
            
    def concrete_classifier(self):
        pass
            
            
            
            
            
            
        