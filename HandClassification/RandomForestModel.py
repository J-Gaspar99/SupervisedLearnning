from Classifier import Classifier
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(Classifier):
    
    def __init__(self, criterion, estimators):
        self.criterion = criterion
        self.estimators = estimators
    
    def concrete_classifier(self):
        classifier = RandomForestClassifier(n_estimators=self.estimators, class_weight='balanced', criterion=self.criterion)
        return classifier