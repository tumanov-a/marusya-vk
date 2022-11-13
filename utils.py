# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pymystem3 import Mystem
from collections import Counter
from razdel import sentenize
from tqdm import tqdm



acceptability_tokenizer = AutoTokenizer.from_pretrained('RussianNLP/ruRoBERTa-large-rucola')
acceptability_model = AutoModelForSequenceClassification.from_pretrained('RussianNLP/ruRoBERTa-large-rucola')

toxicity_model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_checkpoint)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_checkpoint)

resp_qual_tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/response-quality-classifier-large')
resp_qual_model = AutoModelForSequenceClassification.from_pretrained('tinkoff-ai/response-quality-classifier-large')

# device = torch.device('cuda:15')

# if torch.cuda.is_available():
#     toxicity_model.to(device)
#     acceptability_model.to(device)
#     resp_qual_model.to(device)
    
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = toxicity_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(toxicity_model.device)
        proba = torch.sigmoid(toxicity_model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

def preprocess_context(list_of_replics, model_type):
    if model_type == 'bert':
        unk_token = '[UNK]'
        cls_token = '[CLS] '
    elif model_type == 'roberta':
        unk_token = '<unk>'
        cls_token = '<s> '
    elif model_type == 't5':
        unk_token = '<unk>'
        cls_token = ''
    prep_list_of_replics = []
    for replic in list_of_replics:
        user_replic = replic['user'].strip()
        marusia_replic = replic['marusia'].strip()

        if not user_replic:
            user_replic = unk_token
        if not marusia_replic:
            marusia_replic = unk_token
        
        user_replic = '– ' + user_replic
        marusia_replic = '– ' + marusia_replic
        prep_list_of_replics.append(' '.join([user_replic, marusia_replic]))

    whole_dialog = ' '.join(prep_list_of_replics)
    if unk_token == '<unk>':
        whole_dialog = re.sub(r'(– <unk> – <unk>( )*){5}', '– <unk> – <unk> ', whole_dialog)
        whole_dialog = re.sub(r'(– <unk> – <unk>( )*){4}', '– <unk> – <unk> ', whole_dialog)
        whole_dialog = re.sub(r'(– <unk> – <unk>( )*){3}', '– <unk> – <unk> ', whole_dialog)
        whole_dialog = re.sub(r'(– <unk> – <unk>( )*){2}', '– <unk> – <unk> ', whole_dialog)
    elif unk_token == '[UNK]':
        whole_dialog = re.sub(r'(– \[UNK\] – \[UNK\]( )*){5}', '– [UNK] – [UNK] ', whole_dialog)
        whole_dialog = re.sub(r'(– \[UNK\] – \[UNK\]( )*){4}', '– [UNK] – [UNK] ', whole_dialog)
        whole_dialog = re.sub(r'(– \[UNK\] – \[UNK\]( )*){3}', '– [UNK] – [UNK] ', whole_dialog)
        whole_dialog = re.sub(r'(– \[UNK\] – \[UNK\]( )*){2}', '– [UNK] – [UNK] ', whole_dialog)
    return cls_token + whole_dialog


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)

def delete_emoji(context):
    new_context = []
    for turn in context:
        new_turn = {}
        
        new_turn['user'] = re.sub(emoji_pattern, '', turn['user']).strip()
        new_turn['marusia'] = re.sub(emoji_pattern, '', turn['marusia']).strip()
        
        new_context.append(new_turn)
    return new_context


def toxicity_of_context(context):
    toxicities = []
    for x in context:
        user_repl = x['user']
        if user_repl:
            repl_toxicity = text2toxicity(user_repl, True)
            toxicities.append(repl_toxicity)
        else:
            continue
    mean_toxicity = np.mean(toxicities) if toxicities else 0
    return mean_toxicity

def count_ner_in_context(context):
    count_ner = []
    for x in context:
        user_repl = x['user']
        marusia_repl = x['marusia']
        if user_repl:
            repl_toxicity = text2toxicity(user_repl, True)
            toxicities.append(repl_toxicity)
        else:
            continue
    mean_toxicity = np.mean(toxicities) if toxicities else 0
    return mean_toxicity

def predict_acceptability(text):
    inputs = acceptability_tokenizer(text, max_length=128, add_special_tokens=False, return_tensors='pt').to(acceptability_model.device)
    with torch.inference_mode():
        logits = acceptability_model(**inputs).logits
        probas = torch.nn.Softmax(dim=1)(logits)[0].cpu().detach().numpy()
    return probas[0]

def predict_resp_qual(text):
    inputs = resp_qual_tokenizer(text, max_length=128, add_special_tokens=False, return_tensors='pt').to(resp_qual_model.device)
    with torch.inference_mode():
        logits = resp_qual_model(**inputs).logits
        probas = torch.sigmoid(logits)[0].cpu().detach().numpy()
    return ' '.join([str(probas[0]), str(probas[1])])

def return_str_for_resp_qual(df):
    ret = []
    for x in df['context']:
        if len(x['user']) > 0 and len(x['marusia']) > 0:
            ret.append(x['user'])
            ret.append(x['marusia'])

    return'[CLS]' + "[SEP]".join(ret) + '[RESPONSE_TOKEN]' + df['phrase']



del acceptability_tokenizer
del acceptability_model

del toxicity_tokenizer
del toxicity_model

del resp_qual_tokenizer 
del resp_qual_model