import torch

class NERDataset():
    def __init__(self, sentences, kwords, labels, config, data_type='test'):
        self.sentences = sentences
        self.kwords = kwords
        self.labels = labels
        self.config = config
        self.max_length = self.config['MAX_LEN']
        self.tokenizer = self.config['tokenizer']
        self.data_type = data_type
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, id):
        toks = self.config['tokenizer'].tokenize(self.sentences[id])
        label = self.labels[id]

        if len(toks)>self.max_length:
            toks = toks[:self.max_length]
            label = label[:self.max_length]
        
        ########################################
        # Forming the inputs
        ids = self.config['tokenizer'].convert_tokens_to_ids(toks)
        tok_type_id = [0] * len(ids)
        att_mask = [1] * len(ids)
        
        # Padding
        pad_len = self.max_length - len(ids)        
        ids = ids + [2] * pad_len
        tok_type_id = tok_type_id + [0] * pad_len
        att_mask = att_mask + [0] * pad_len
        
        ########################################            
        # Forming the label
        if self.data_type !='test':
            label = label + [2] * pad_len
        else:
            label = 1
            
        return {'ids': torch.tensor(ids, dtype = torch.long),
                'tok_type_id': torch.tensor(tok_type_id, dtype = torch.long),
                'att_mask': torch.tensor(att_mask, dtype = torch.long),
                'target': torch.tensor(label, dtype = torch.long)
               }
