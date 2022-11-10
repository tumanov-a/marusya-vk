from models.models import *

model_params = {
            'bert': {
                    'hidden_neurons': 512
                    },
            't5': {
                    'hidden_neurons': 512
                    }
            }

def create_model(model_name, loss):
    if loss == 'bce' or loss == 'soft':
        n_labels = 1
    elif loss == 'ce':
        n_labels = 2

    if model_name == 'bert':
        model = BertClassifier(hidden_neurons=model_params[model_name]['hidden_neurons'], n_labels=n_labels, loss=loss)
    elif model_name == 't5':
        model = T5Classifier(hidden_neurons=model_params[model_name]['hidden_neurons'], n_labels=n_labels, loss=loss)
    return model
