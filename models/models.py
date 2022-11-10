import torch
from transformers import AutoConfig, AutoModel, T5Config, T5Model

class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_neurons, n_labels):
        super(BertClassifier, self).__init__()
        self.n_labels = n_labels
        bert_config = AutoConfig.from_pretrained(model_name, num_labels=self.n_labels)
        self.model = AutoModel.from_pretrained(model_name, config=bert_config)
        self.hidden_neurons = hidden_neurons
        self.linear1 = torch.nn.Linear(bert_config.hidden_size, int(self.hidden_neurons / 2))
        self.classifier = torch.nn.Linear(bert_config.hidden_size, self.n_labels)
        self.dropout = torch.nn.Dropout(.3)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(bert_config.hidden_size, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = bert_output[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        scores = self.classifier(feature)
        return scores


class T5Classifier(torch.nn.Module):
    def __init__(self, hidden_neurons, n_labels):
        super(T5Classifier, self).__init__()
        config = T5Config.from_pretrained('sberbank-ai/ruT5-large')
        self.model = T5Model.from_pretrained('sberbank-ai/ruT5-large', config=config).encoder
        self.hidden_neurons = hidden_neurons
        self.n_labels = n_labels
        self.linear1 = torch.nn.Linear(config.hidden_size, self.hidden_neurons)
        #self.linear2 = torch.nn.Linear(self.hidden_neurons, int(self.hidden_neurons / 2))
        #self.classifier = torch.nn.Linear(int(self.hidden_neurons / 2), self.n_target)
        self.classifier = torch.nn.Linear(self.hidden_neurons, self.n_labels)
        #self.activation = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(.3)

    def forward(self, input_ids, attention_mask):
        t5_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = t5_output[0]
        pooled_output = seq_output.mean(axis=1)
        #pooled_output = self.dropout(pooled_output)
        x1 = self.dropout(self.linear1(pooled_output))
        #x1 = self.dropout(self.linear2(x1))
        scores = self.classifier(x1)
        #scores = self.activation(scores)
        return scores