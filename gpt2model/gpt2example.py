import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import os
from models import GPT2
gpt2_ckpt_dir = '/home/xiaoguzai/下载/CPM_LM_2.6B_TF/'
gpt2_ckpt_file = gpt2_ckpt_dir + "model.ckpt"
gpt2_config_file = gpt2_ckpt_dir + "config.json"
gpt2_spm_path = gpt2_ckpt_dir + "chinese_vocab.model"
import jieba

from tokenizers2 import FullTokenizer
tokenizer = FullTokenizer(gpt2_spm_path)
def convert_list_to_str(lists):
    results = ''
    for item in lists:
        results = results+item
    return  results

query = u"""美国的首都是华盛顿
法国的首都是巴黎
日本的首都是东京
中国的首都是"""
token_ids = tokenizer.tokenize(query)
print('token_ids = ')
print(token_ids)
results = tokenizer.convert_ids_to_tokens(token_ids)
#results = convert_list_to_str(results)
#!!!目前这个分词的部分内容还有待于修正!!!

result_ids = [356, 11, 6597, 15, 13176, 8, 3, 1857, 11, 6597, 15, 6040, 8, 3, 461, 11, 6597, 15, 5233, 8, 3, 98, 11, 6597, 15]
print('results = ')
print(results)

steps = 1
border = 0.999
from loader import load_stock_weights
def predict(token_ids,border):
    maxlen = 25
    input_ids = keras.layers.Input(shape=(maxlen,),dtype='int32',name="input_ids")
    gpt2model = GPT2(mlm=True)
    outputs = gpt2model(input_ids)
    load_stock_weights(gpt2model,gpt2_ckpt_file)
    input_ids = result_ids
    results = gpt2model([input_ids])
    results = results.numpy()
    probas = np.array([results[0][-1]])
    #之前这里的probas = list(results[0][maxlen-1])出现错误
    #感觉应该是maxlen值不对
    print('%%%probas = %%%')
    np.set_printoptions(threshold=np.inf)
    probas /= probas.sum(axis=-1,keepdims=True)
    np.set_printoptions(threshold=np.inf) #全部输出
    #print('probas1 = ')
    #print(probas)
    #得到的结果为(1,30000)的对应权重矩阵
    p_indices = probas.argsort(axis=1)[:,::-1]
    #概率逆序排序,并记录下相应的标记
    #!!!!!!到这里的p_indices都没有出现NaN的具体标志!!!!!!
    probas = np.take_along_axis(probas,p_indices,axis=1)
    np.set_printoptions(threshold=50000)
    #print('probas2 = ')
    #print(probas)
    cumsum_probas = np.cumsum(probas,axis=1)
    #概率与后面的概率进行列相加
    print('cumsum_probas = ')
    print(cumsum_probas)
    flag = cumsum_probas>border
    probas[flag] = 0
    #将后面概率小的并且超过border位置的概率值标记为0
    probas /= probas.sum(axis=1,keepdims=True)
    #进行归一化操作
    print('((((((((probas = ))))))))')
    print(probas)
    np.set_printoptions(threshold=50000)
    #print('probas3 = ')
    #print(probas)
    sample_func = lambda p:np.random.choice(len(p),p=p)
    sample_ids = np.apply_along_axis(sample_func,1,probas)
    #对现有的概率进行随即取样
    print('sample_ids = ')
    print(sample_ids)
    return [p_indices[0][sample_ids[0]]]

def convert_list_to_str(lists):
    results = ''
    for item in lists:
        results = results+item
    return  results

for i in range(steps):
    print('******i = ******')
    print(i)
    results = predict(token_ids,border)
    print('######results = ######')
    print(results)
    token_ids = np.concatenate((token_ids,results),axis=0)
results = tokenizer.convert_ids_to_tokens(token_ids)
results = convert_list_to_str(results)
print('results = ')
print(results)