import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
from loader import load_stock_weights
import json
import math
from models import Bert
import numpy as np
bert_ckpt_dir="/home/xiaoguzai/下载/chinese-base/"
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
tokens = tokenizer.tokenize(u'科学技术是第一生产力')
tokens = ['[CLS]']+tokens+['[SEP]']
token_ids = tokenizer.convert_tokens_to_ids(tokens)

length = len(token_ids)
input_ids = keras.layers.Input(shape=(length,),dtype='int32',name='input_ids')
outputs = bertmodel(input_ids)
load_stock_weights(bertmodel,bert_ckpt_file)

text = ['[MASK]']
mask_ids = tokenizer.convert_tokens_to_ids(text)
token_mask_id = mask_ids[0]
token_ids[3] = token_ids[4] = token_mask_id
probas = bertmodel(np.array([token_ids]))
print(tokenizer.convert_ids_to_tokens(np.argmax(probas[0][3:5],axis=1)))