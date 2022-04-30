from FileHandling import FileHandling
from NeuralNetworkModel import NeuralNetworkModel
from RandomForestModel import RandomForestModel
from KNNModel import KNNModel
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%
#учитавање података

fh = FileHandling()
paths_of_images = fh.create_file_list('slike_baza')
x_raw = fh.load_images(paths_of_images)

#%%
splitted = paths_of_images['Images paths'].str.split('_', expand=True)
splitted.drop([0, 3], axis=1, inplace=True)
splitted.columns = ['Set', 'Class', 'Image id']

#%%
y = splitted.loc[:, 'Class']
classes = splitted.loc[:, 'Class'].unique()

groupped = splitted.groupby('Set').count()
class_distribution = groupped['Class']


#%%
# Функција за разлагање главних компоненти(eng. Principal component analysis - PCA) 
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

#%%
# Модели неуралне мреже

pca_components = [60, 70, 80, 90, 100, 130, 140, 150]
hidden_layers = [(60,80),(60, 80, 120),(100, 200), (128, 128),(128, 128, 128)]
results_nnm = pd.DataFrame(columns=['Komponente', 'Slojevi', 'Optimizacioni alg.', 'Aktivaciona f-ja', 'Tacnost', 'Makro osetljivost', 'Mikro osetljivost', 'Makro F-mera', 'Mikro F-mera'])
solvers = ['adam', 'lbfgs', 'sgd']
activations = ['logistic', 'tanh']

for pca_component in pca_components:
    x_reduced = pca_reduction(x_raw, pca_component)
    for hidden_layer in hidden_layers:
        for solver in solvers:
            for activation in activations:
                nnm = NeuralNetworkModel(activation, solver, hidden_layer)
                fin_conf_mat = nnm.start(x_reduced, y, classes, class_distribution)
                results_temp = nnm.calculate_metrics(fin_conf_mat, classes)
                results_temp['Komponente'] = pca_component
                results_temp['Slojevi'] = hidden_layer
                results_temp['Optimizacioni alg.'] = solver
                results_temp['Aktivaciona f-ja'] = activation
                
                results_nnm = results_nnm.append(results_temp, ignore_index=True)
                

#%%
# коначни модел неуралне мреже
nnm = NeuralNetworkModel('logistic', 'adam', (128, 128))
x_reduced = pca_reduction(x_raw, 100)
fin_conf_mat = nnm.final_model(x_reduced, y, classes, class_distribution)
results_temp = nnm.calculate_metrics(fin_conf_mat, classes)

#%%
# модели случајне шуме
criterions = ['gini', 'entropy']
estimators = [75, 100, 125]
pca_components = [60, 70, 80, 90, 100]
results_rf = pd.DataFrame(columns=['Criterion', 'Estimators', 'Components', 'Tacnost', 'Makro osetljivost', 'Mikro osetljivost', 'Makro preciznost', 'Mikro preciznost', 'Specificnost', 'Makro F-mera', 'Mikro F-mera'])
i = 0

for pca_component in pca_components:
    x_reduced = pca_reduction(x_raw, pca_component)
    for estimator in estimators:
        for criterion in criterions:        
            rf = RandomForestModel(criterion, estimator)
            fin_conf_mat = rf.start(x_reduced, y, classes, class_distribution)
            results_temp = rf.calculate_metrics(fin_conf_mat, classes)
            results_temp['Components'] = pca_component
            results_temp['Criterion'] = criterion
            results_temp['Estimators'] = estimator
            print('BROJ ITERACIJE: ', i)
            i += 1
            results_rf = results_rf.append(results_temp, ignore_index=True)


#%%
# Коначни модел случајне шуме
rf = RandomForestModel('entropy', 75)
x_reduced = pca_reduction(x_raw, 80)
fin_conf_mat = rf.final_model(x_raw, y, classes, class_distribution)
results_temp = rf.calculate_metrics(fin_conf_mat, classes)

#%%
# Модели К најближих суседа

neighbors = range(1,50)
pca_components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
weights = ['uniform', 'distance']
metrics = ['euclidean','chebyshev','manhattan']   

results_knn = pd.DataFrame(columns=['Neighbors','Weight','Metric' ,'Components', 'Tacnost', 'Makro osetljivost', 'Mikro osetljivost', 'Makro preciznost', 'Mikro preciznost', 'Specificnost', 'Makro F-mera', 'Mikro F-mera', 'Error_rate'])
i = 0

for pca_component in pca_components:
    x_reduced = pca_reduction(x_raw, pca_component)
    for neighbor in neighbors:
      for metric in metrics:
        for weight in weights:
                knn = KNNModel(neighbor, metric, weight)
                fin_conf_mat, error_rate = knn.start(x_reduced, y, classes, class_distribution)
                results_temp = knn.calculate_metrics(fin_conf_mat, classes)
                results_temp['Neighbors'] = neighbor
                results_temp['Components'] = pca_component
                results_temp['Weight'] = weight
                results_temp['Metric'] = metric
                results_temp['Error_rate'] = error_rate
                results_knn = results_knn.append(results_temp, ignore_index=True)
                print('BROJ ITERACIJE: ', i+1)
                i += 1
           

#%%
# Коначни модел к најближих суседа
knn = KNNModel(1,'manhattan','distance')
x_reduced = pca_reduction(x_raw, 80)
fin_conf_mat = knn.final_model(x_reduced, y, classes, class_distribution)
results_temp = knn.calculate_metrics(fin_conf_mat, classes)




