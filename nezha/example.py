nezha_ckpt_dir="/home/xiaoguzai/下载/NEZHA-Base-WWM/"
nezha_ckpt_file = nezha_ckpt_dir + "model.ckpt-691689"
nezha_config_file = nezha_ckpt_dir + "bert_config.json"

import json
with open(nezha_config_file,'r') as load_f:
    config = json.load(load_f)
print('config = ')
print(config)
config['embedding_size'] = config['hidden_size']
config['num_layers'] = config['num_hidden_layers']

from nezha import Bert
import tensorflow as tf
import tensorflow.keras as keras
batch_size = 48
max_seq_len = 512
nezhamodel = Bert(maxlen=max_seq_len,batch_size=batch_size,**config)
input_ids = keras.layers.Input(shape=(None,),dtype='int32',name="input_ids")
result = nezhamodel(input_ids)
#!!!名字改变原因：必须用Input_ids过一遍,名称才会发生对应的改变

from nezha_loader import load_nezha_stock_weights
load_nezha_stock_weights(nezhamodel,nezha_ckpt_file)

input_ids = tf.ones((32,512))
results = nezhamodel(input_ids)
print('...results = ...')
print(results)
