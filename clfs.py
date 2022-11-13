import torch
from transformers import AutoConfig, AutoModel, T5Config, T5Model

class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_neurons, feature_size, n_labels, loss_type):
        super(BertClassifier, self).__init__()
        self.n_labels = n_labels
        self.feature_size = feature_size
        bert_config = AutoConfig.from_pretrained(model_name, num_labels=self.n_labels)
        self.model = AutoModel.from_pretrained(model_name, config=bert_config)
        self.hidden_neurons = hidden_neurons
        self.classifier = torch.nn.Linear(bert_config.hidden_size + 1, self.n_labels)
        self.dropout = torch.nn.Dropout(.1)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(bert_config.hidden_size, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1),
            torch.nn.Softmax(dim=1)
        )
        self.attention_features = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
            torch.nn.Softmax(dim=1)
        )
        if n_labels == 1 and loss_type == 'bce':
            self.activation = torch.nn.Sigmoid()
        elif n_labels == 1 and loss_type == 'soft':
            self.activation = torch.nn.Tanh()
        
    def forward(self, input_ids, attention_mask, features):
        bert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = bert_output[0]
        weights = self.attention(last_hidden_states)
        weights_features = self.attention_features(features)
        relu_features = torch.nn.ReLU()(features)
        feature_1 = torch.sum(weights * last_hidden_states, dim=1)
        feature_2 = torch.sum(weights_features * relu_features, dim=1)
        feature = torch.cat([feature_1, feature_2.unsqueeze(1)], axis=1)
        scores = self.dropout(self.classifier(feature))
        scores = self.classifier(feature)
        if self.n_labels == 1:
            scores = self.activation(scores)
        return scores


class T5Classifier(torch.nn.Module):
    def __init__(self, model_name, hidden_neurons, n_labels, loss_type):
        super(T5Classifier, self).__init__()
        config = T5Config.from_pretrained(model_name)
        self.model = T5Model.from_pretrained(model_name, config=config).encoder
        self.hidden_neurons = hidden_neurons
        self.n_labels = n_labels
        self.linear1 = torch.nn.Linear(config.hidden_size, self.hidden_neurons)
        #self.linear2 = torch.nn.Linear(self.hidden_neurons, int(self.hidden_neurons / 2))
        #self.classifier = torch.nn.Linear(int(self.hidden_neurons / 2), self.n_target)
        self.classifier = torch.nn.Linear(self.hidden_neurons, self.n_labels)
        #self.activation = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(.3)
        if n_labels == 2:
            self.activation = torch.nn.Sigmoid()
        elif n_labels == 1 and loss_type == 'bce':
            self.activation = torch.nn.Sigmoid()
        elif n_labels == 1 and loss_type == 'soft':
            self.activation = torch.nn.Tanh()

    def forward(self, input_ids, attention_mask):
        t5_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = t5_output[0]
        pooled_output = seq_output.mean(axis=1)
        #pooled_output = self.dropout(pooled_output)
        x1 = self.dropout(self.linear1(pooled_output))
        #x1 = self.dropout(self.linear2(x1))
        scores = self.classifier(x1)
        scores = self.activation(scores)
        return scores