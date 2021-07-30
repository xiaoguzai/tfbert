import tensorflow as tf
import json
import tensorflow.keras.backend as K
import re
import tensorflow.keras as keras
import numpy as np
from tqdm import tqdm
config_path = '/home/xiaoguzai/数据/unilm/mixed_corpus_bert_base_model/bert_config.json'
checkpoint_path = '/home/xiaoguzai/数据/unilm/mixed_corpus_bert_base_model/bert_model.ckpt'
dict_path = '/home/xiaoguzai/数据/unilm/mixed_corpus_bert_base_model/vocab.txt'

def is_equal(a,b):
    a = round(float(a),6)
    b = round(float(b),6)
    return a == b

def remove_bucket(equation):
    """去掉冗余的括号
    """
    l_buckets, buckets = [], []
    for i, c in enumerate(equation):
        if c == '(':
            l_buckets.append(i)
        elif c == ')':
            buckets.append((l_buckets.pop(), i))
    eval_equation = eval(equation)
    for l, r in buckets:
        new_equation = '%s %s %s' % (
            equation[:l], equation[l + 1:r], equation[r + 1:]
        )
        try:
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')
#回头可以实验一下不去掉冗余的括号以及不将空格替换为''的变化

def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    for l in open(filename):
        l = json.loads(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        # 处理带分数,re.sub:实现正则的替换
        # ()标记一个子表达式的开始和结束位置，子表达式可以获取供以后使用
        # 要匹配这些字符，请使用\(和\),所以说(\d+)后面的\(代表着左半边的括号
        # (\d+/\d+)后面的\)代表着右半部分的括号
        r"""
        整个替换的过程：对于a%统一替换为(a/100),
        对于a(b/c),统一替换为(a+b/c),
        对于(a/b),去掉括号变为a/b,
        对于比例的冒号,统一替换为/。
        """
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        #3(1/2)->(3+1/2)
        
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        #38(1/2)->(38+1/2),(38(1/2))->(38+(1/2))
        
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # question这里不去除分号？感觉去除分号可能更好一点
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        #!!!小细节部分，前面的'x='需要被去掉
        try:
            if is_equal(eval(equation), eval(answer)):
                D.append((question, remove_bucket(equation), answer))
        except:
            #如果eval(equation)和eval(answer)的结果不一致的情况下
            #视为无效公式，不放入对应的D数组中
            continue
    return D

# 加载数据集
train_data = load_data('/home/xiaoguzai/数据/data/train.ape.json')
valid_data = load_data('/home/xiaoguzai/数据/data/valid.ape.json')
test_data = load_data('/home/xiaoguzai/数据/data/test.ape.json')

from tokenization import load_vocab
#录入权重的时候还是需要正常切词并录入权重
#只不过输出的时候权重词典变得精简了
token_dict = load_vocab(dict_path)
print('token_dict = ')
print(token_dict)

startswith = ['[PAD]','[UNK]','[CLS]','[SEP]']
new_token_dict,keep_tokens = {},[]
for t in startswith:
    new_token_dict[t] = len(new_token_dict)
    keep_tokens.append(token_dict[t])

from tokenization import FullTokenizer
for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
    if t not in new_token_dict:
        keep = True
        if len(t) > 1:
            for c in FullTokenizer.stem(t):
                if (
                    FullTokenizer._is_cjk_character(c) or
                    FullTokenizer._is_punctuation(c)
                ):
                    keep = False
                    break
        #注意单独使用!和前面带有前缀##!时的keep对应值不同
        #单独使用!不满足if len(t) > 1,直接进入下面的if keep
        #条件判定，而如果##!满足len(t) > 1,就会进行条件判定
        #
        if keep:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])
r"""new_token_dict = {'[PAD]':0,'[UNK]':1,'[CLS]':2,
'[SEP]':3,'!':4,'"':5,'#':6,...,'##😎': 13583}
,len(keep_tokens) = 13584,len(new_token_dict) = 13584
"""

print('new_token_dict = ')
print(new_token_dict)

from models import Bert
from models import Embeddings
import tensorflow.keras as keras

tokenizer = FullTokenizer(vocab_file=new_token_dict)
#使用新的new_token_dict对语句进行分词切分
token_ids = []
segment_ids = []
for data in train_data:
    text1 = data[0]
    text2 = data[1]
    token1 = tokenizer.tokenize(text1)
    token2 = tokenizer.tokenize(text2)
    tokens = ["[CLS]"]+token1+["[SEP]"]+token2+["[SEP]"]
    token_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids.append(token_id)
    segment1 = [0]*(len(token1)+2)
    segment2 = [1]*(len(token2)+1)
    segment_id = segment1+segment2
    segment_ids.append(segment_id)

class CrossEntropy(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(CrossEntropy,self).__init__(**kwargs)
    
    def compute_loss(self,inputdata,y_pred):
        y_true = inputdata[0]
        y_mask = inputdata[0]
        loss = K.sparse_categorical_crossentropy(y_true,y_pred)
        y_mask = tf.cast(y_mask,dtype=tf.float32)
        loss = K.sum(tf.multiply(y_mask,loss))/K.sum(y_mask)
        #想要将前面的单词内容mask掉
        return loss
    
    def call(self,inputs):
        y_pred = inputs[1]
        inputdata = inputs[0]
        loss = self.compute_loss(inputdata,y_pred)
        self.add_loss(loss,inputs=inputs)
        return inputs

import json
json_file = '/home/xiaoguzai/数据/unilm/mixed_corpus_bert_base_model/bert_config.json'
with open(json_file,'r') as load_f:
    load_dict = json.load(load_f)
    load_dict['hidden_dropout'] = load_dict['attention_probs_dropout_prob']
    load_dict['num_layers'] = load_dict['num_hidden_layers']
    load_dict['pooler_num_fc_layers'] = load_dict['pooler_fc_size']
    load_dict['embedding_size'] = load_dict['hidden_size']
    load_dict['vocab_size'] = len(new_token_dict)
    load_dict['embedding_size'] = load_dict['hidden_size']
    print(load_dict)

batch_size = 5
max_seq_len = 128
bertmodel = Bert(maxlen=max_seq_len,with_mlm=True,mode='unilm',
                solution='seq2seq',new_tokens=new_token_dict,**load_dict)
input_ids = [keras.layers.Input(shape=(None,),dtype='int32',name="token_ids"),
            keras.layers.Input(shape=(None,),dtype='int32',name="segment_ids")]
output = bertmodel(input_ids)
#使用一个bertmodeldata进行测试内容
#output = KerasTensor(shape=(None,128,30522),dtype=tf.float32)
output = CrossEntropy()([input_ids,output])
#返回的还是常规的input
model = keras.models.Model(input_ids,output)
#上面内容使用seq2seq循环构建模型计算相应的损失内容
model.compile(optimizer=keras.optimizers.Adam())

#!!!!!!!!!!!!!!!!!!!!!这里应该使用keras.models.Model(model.inputs,output)
#否则input_ids相当于被固定好的形状，在这里使用肯定不行，在model.fit训练的过程中
#会被相应的报错，所以这里需要根据输入进行调整

from loader import load_stock_weights
load_stock_weights(bert=bertmodel,new_tokens=keep_tokens,ckpt_path=checkpoint_path)

def sequence_padding(inputs,padding = 0):
    length = max([len(x) for x in inputs])
    pad_width = [(0,0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0,length-len(x))
        x = np.pad(x,pad_width,'constant',constant_values=padding)
        outputs.append(x)
    return outputs

class DataGenerator(object):
    def __init__(self,token_ids,segment_ids,batch_size=32,maxlen=128):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.batch_size = batch_size
        self.steps = int(np.floor(len(self.token_ids)/self.batch_size))
        self.totals = len(self.token_ids)
        self.maxlen = maxlen
    
    def __len__(self):
        return int(np.floor(len(self.token_ids)/self.batch_size))
    
    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        indices = list(range(len(self.token_ids)))
        np.random.shuffle(indices)
        for i in indices:
            yield self.token_ids[i],self.segment_ids[i]
        
    def __iter__(self,random=False):
        random = False
        batch_data = []
        batch_token_ids,batch_segment_ids = [],[]
        currents = 0
        for token_ids,segment_ids in self.sample(random):
        #传入的数据在下面定义train_generator = data_generator(train_data, batch_size)
        #这里如果使用tqdm(self.sample(random))，它就会连续地不断产生红色区域
        #如果不使用tqdm(self.sample(random))，它就会连续以...的形式输出进度
        #因为model.fit()函数之中自带相应的进度条
            if len(token_ids) > self.maxlen:
                token_ids = token_ids[:self.maxlen]
                segment_ids = segment_ids[:self.maxlen]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            currents = currents+1
            if len(batch_token_ids) == self.batch_size or currents == self.totals:
                #len(batch_token_ids) == self.batch_size:当前批次结束
                #is_end:所有数据结束(可能不够一个批次)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [np.array(batch_token_ids),np.array(batch_segment_ids)]
                r"""
                这里的batch_token_ids和batch_segment_ids外面必须加上np.array
                """
                batch_token_ids, batch_segment_ids = [], []
                batch_data = []
                #每一个批次结束的时候

    def cycle(self,random=True):
        while True:
            for d in self.__iter__(random):
                yield d
r"""
这里返回的d = ([array([[2,656,...,20,3],[2,2105,...,0,0],
...[2,569,...,0,0]]),array([[0,0,...,0,0,1,1],[0,0,...,0,0],
...[0,0,...,0,0]])],None)
这里的前面一个属于x，以一个元组的形式输出，后面一个None为y，因为这里
是seq2seq，所以不需要y

如果使用yield ([batch_token_ids,batch_segment_ids],None)的时候
报错tuple index out of range
如果使用yield ([np.array(batch_token_ids),np.array(batch_segment_ids)
,None])的时候，会报错  (0) Invalid argument:  required broadcastable shapes at loc(unknown)
 [[node model/bert/embeddings/add (defined at 
/home/xiaoguzai/代码/unilm-main/models.py:296) ]]
"""

from Evaluator import Evaluator

train_generator = DataGenerator(token_ids,segment_ids,batch_size=32,maxlen=128)
topk = 3
evaluator = Evaluator(topk=topk,data=valid_data,model=model)
evaluator.on_epoch_end()
