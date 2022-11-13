from clfs import BertClassifier, T5Classifier

model_params = {
            'bert': {
                    'hidden_neurons': 512,
                    'hugg_name': 'sberbank-ai/ruBert-large'
                    },
            't5': {
                    'hidden_neurons': 512,
                    'hugg_name': 'sberbank-ai/ruT5-large'
                    },
            'roberta': {
                    'hidden_neurons': 512,
                    'hugg_name': 'DeepPavlov/xlm-roberta-large-en-ru'
                    },
            'roberta-ru': {
                    'hidden_neurons': 512,
                    'hugg_name': 'DeepPavlov/xlm-roberta-large-en-ru'
                    }
            }

def create_model(model_name, loss_type, n_features):
    if loss_type == 'bce' or loss_type == 'soft':
        n_labels = 1
    elif loss_type == 'ce':
        n_labels = 2

    if model_name == 'bert':
        model = BertClassifier(model_name=model_params[model_name]['hugg_name'], hidden_neurons=model_params[model_name]['hidden_neurons'], feature_size=n_features, n_labels=n_labels, loss_type=loss_type)
    elif model_name == 'roberta':
        model = BertClassifier(model_name=model_params[model_name]['hugg_name'], hidden_neurons=model_params[model_name]['hidden_neurons'], feature_size=n_features, n_labels=n_labels, loss_type=loss_type)
    elif model_name == 'roberta-ru':
        model = BertClassifier(model_name=model_params[model_name]['hugg_name'], hidden_neurons=model_params[model_name]['hidden_neurons'], feature_size=n_features, n_labels=n_labels, loss_type=loss_type)
    elif model_name == 't5':
        model = T5Classifier(model_name=model_params[model_name]['hugg_name'], hidden_neurons=model_params[model_name]['hidden_neurons'], n_labels=n_labels, loss_type=loss_type)
    return model
