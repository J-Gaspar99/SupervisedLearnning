from Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNNModel(Classifier):
    
    def __init__(self, neighbors, metric, weights):
        self.neighbors = neighbors
        self.metric = metric
        self.weights = weights
       # self.algorithm = algorithm
    
    def concrete_classifier(self):
        classifier = KNeighborsClassifier(n_neighbors=self.neighbors, metric=self.metric, weights=self.weights, algorithm='auto')
        return classifier