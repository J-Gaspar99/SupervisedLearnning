from Classifier import Classifier
from sklearn.neural_network import MLPClassifier

class NeuralNetworkModel(Classifier):
    
    def __init__(self, activation, solver, hidden_layer_structure ):
        self.hidden_layer_structure = hidden_layer_structure
        self.activation = activation
        self.solver = solver
        
    def concrete_classifier(self):
        classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_structure, activation=self.activation,
                      solver=self.solver, batch_size=50, learning_rate='constant', 
                      learning_rate_init=0.001, max_iter=500, shuffle=True,
                      random_state=42, early_stopping=True, n_iter_no_change=10,
                      validation_fraction=0.1, verbose=False)
        return classifier