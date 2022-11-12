import torch
import numpy as np

class Dataset():
    def __init__(self, sentences, labels, config, features=None):
        self.sentences = sentences
        self.labels = labels
        self.features = features
        self.config = config
        self.max_length = self.config['MAX_LEN']
        self.tokenizer = self.config['tokenizer']
        self.model = self.config['model']
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, id):
        toks = self.config['tokenizer'].tokenize(self.sentences[id])
        label = self.labels[id]
        # if isinstance(self.features, np.ndarray):
        feature = self.features[id]

        if len(toks) > self.max_length:
            if self.model != 't5':
                toks = toks[- self.max_length + 1:]
                toks = [self.config['tokenizer'].cls_token] + toks
            else:
                toks = toks[-self.max_length:]
        
        ########################################
        # Forming the inputs
        ids = self.config['tokenizer'].convert_tokens_to_ids(toks)
        att_mask = [1] * len(ids)
        
        # Padding
        pad_len = self.max_length - len(ids)        
        ids = ids + [0] * pad_len
        att_mask = att_mask + [0] * pad_len
        # if isinstance(self.features, np.ndarray):
        return {'ids': torch.tensor(ids, dtype = torch.long),
                'att_mask': torch.tensor(att_mask, dtype = torch.long),
                'features': torch.tensor(feature, dtype = torch.float),
                'target': torch.tensor(label)
            }
        # else:
        #     return {'ids': torch.tensor(ids, dtype = torch.long),
        #             'att_mask': torch.tensor(att_mask, dtype = torch.long),
        #             'target': torch.tensor(label)
        #         }
