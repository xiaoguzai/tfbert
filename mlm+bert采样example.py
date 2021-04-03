import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from loader import load_stock_weights
import json
import math
from models import Bert
import numpy as np
bert_ckpt_dir="/home/xiaoguzai/下载/chinese_L-12_H-768_A-12/"
bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
bert_config_file = bert_ckpt_dir + "bert_config.json"
with open(bert_config_file) as f:
    config = json.load(f)
config['with_mlm'] = True
config['batch_size'] = 1
bertmodel = Bert(**config)

from tokenization import FullTokenizer
import os
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
init_sent = '北京新增3例本地确诊病例和1例无症状感染者'#or None
minlen,maxlen = 8,32
steps = 100
max_seq_len = 0
coverged_steps = 1000
vocab_size = config['vocab_size']
#tokens = tokenizer.tokenize(init_sent)
#token_ids = tokenizer.convert_tokens_to_ids(tokens)
text = ['[MASK]']
mask_ids = tokenizer.convert_tokens_to_ids(text)
token_mask_id = mask_ids[0]
if init_sent == None:
    length = np.random.randint(minlen,maxlen+2)
    input_ids = keras.layers.Input(shape = (length,),dtype='int32',name='input_ids')
    outputs = bertmodel(input_ids)
    load_stock_weights(bertmodel,bert_ckpt_file)
    tokens = ['[CLS]']+['[MASK]']*(length-2)+['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
else:
    tokens = tokenizer.tokenize(init_sent)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    length = len(token_ids)
    input_ids = keras.layers.Input(shape = (length,),dtype='int32',name='input_ids')
    outputs = bertmodel(input_ids)
    load_stock_weights(bertmodel,bert_ckpt_file)
sentences = []
for  _  in  tqdm(range(steps),desc='Sampling'):
    i = np.random.choice(length)
    token_ids[i] = token_mask_id
    probas = bertmodel(np.array([token_ids]))[0,i]
    probas = np.array(probas)
    probas /= probas.sum()
    token = np.random.choice(vocab_size,p=probas)
    token_ids[i] = token
    sentences.append(''.join(tokenizer.convert_ids_to_tokens(token_ids)))
print('采样结果为')
print(sentences)