import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
bert_ckpt_dir="/home/xiaoguzai/origin-code/uncased_L-12_H-768_A-12/"
bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
bert_config_file = bert_ckpt_dir + "bert_config.json"

import re
import pandas as pd
import numpy as np
import os
data = {}
data["sentence"] = []
data["sentiment"] = []
for file_path in tqdm(os.listdir('/home/xiaoguzai/origin-code/aclImdb/train/pos'),desc=os.path.basename('/home/xiaoguzai/origin-code/aclImdb/train/pos')):
    with tf.io.gfile.GFile(os.path.join('/home/xiaoguzai/origin-code/aclImdb/train/pos',file_path),"r") as f:
        data["sentence"].append(f.read())
        data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        #data["sentence"]存储原文，比如0_9.txt,则后面对应的情感态度内容就为9
pos_df = pd.DataFrame.from_dict(data)
data = {}
data["sentence"] = []
data["sentiment"] = []
for file_path in tqdm(os.listdir('/home/xiaoguzai/origin-code/aclImdb/train/neg'),desc=os.path.basename('/home/xiaoguzai/origin-code/aclImdb/train/neg')):
    with tf.io.gfile.GFile(os.path.join('/home/xiaoguzai/origin-code/aclImdb/train/neg',file_path),"r") as f:
        data["sentence"].append(f.read())
        data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
neg_df = pd.DataFrame.from_dict(data)
pos_df["polarity"] = 1
neg_df["polarity"] = 0
train_df = pd.concat([pos_df,neg_df]).sample(frac=1).reset_index(drop=True)

import pandas as pd
import numpy as np
data = {}
data["sentence"] = []
data["sentiment"] = []
for file_path in tqdm(os.listdir('/home/xiaoguzai/origin-code/aclImdb/test/pos'),desc=os.path.basename('/home/xiaoguzai/origin-code/aclImdb/test/pos')):
    with tf.io.gfile.GFile(os.path.join('/home/xiaoguzai/origin-code/aclImdb/test/pos',file_path),"r") as f:
        data["sentence"].append(f.read())
        data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        #data["sentence"]存储原文，比如0_9.txt,则后面对应的情感态度内容就为9
pos_df = pd.DataFrame.from_dict(data)
data = {}
data["sentence"] = []
data["sentiment"] = []
for file_path in tqdm(os.listdir('/home/xiaoguzai/origin-code/aclImdb/test/neg'),desc=os.path.basename('/home/xiaoguzai/origin-code/aclImdb/test/neg')):
    with tf.io.gfile.GFile(os.path.join('/home/xiaoguzai/origin-code/aclImdb/test/neg',file_path),"r") as f:
        data["sentence"].append(f.read())
        data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
neg_df = pd.DataFrame.from_dict(data)
pos_df["polarity"] = 1
neg_df["polarity"] = 0
test_df = pd.concat([pos_df,neg_df]).sample(frac=1).reset_index(drop=True)



train, test = map(lambda df: df.reindex(df["sentence"].str.len().sort_values().index), 
                  [train_df, test_df])
sample_size = 2560
train, test = train.head(sample_size),test.head(sample_size)
max_seq_len = 128

from tokenization import FullTokenizer
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
def _prepare(df):
#将前后的位置
    global max_seq_len
    x, y = [], []
    with tqdm(total=df.shape[0], unit_scale=True) as pbar:
        for ndx, row in df.iterrows():
            text, label = row["sentence"], row["polarity"]
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            max_seq_len = max(max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(int(label))
            pbar.update()
    return np.array(x), np.array(y)
def _pad(ids):
    global max_seq_len
    x = []
    token_type_ids = [0] * max_seq_len
    for input_ids in ids:
        input_ids = input_ids[:min(len(input_ids), max_seq_len - 2)]
        #留两个位置为["CLS"]和["SEP"]标志的位置
        input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
        x.append(np.array(input_ids))
        #t.append(token_type_ids)
    return np.array(x)#np.array(t)
((train_x,train_y),(test_x,test_y)) = map(_prepare,[train,test])
((train_x),(test_x)) = map(_pad,[train_x,test_x])


import models
from models import Bert
from models import Embeddings
batch_size = 24
bertmodel = Bert(maxlen=max_seq_len,batch_size=batch_size)


input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
#segment_ids    = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
#position_ids   = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
output = bertmodel(input_ids)
cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
cls_out = keras.layers.Dropout(0.5)(cls_out)
logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
logits = keras.layers.Dropout(0.5)(logits)
logits = keras.layers.Dense(units=2, activation="softmax")(logits)
model1 = keras.Model(inputs=input_ids,outputs=logits)
model = model1
model.build(input_shape=(None,max_seq_len))
model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
#这里的评价函数用于评估当前训练模型的性能，
#当模型编译后，评价函数应该作为metrics的参数来输入
#评价函数和损失函数相似，只不过评价函数的结果不会用于训练过程中
model.summary()


from loader import load_stock_weights
load_stock_weights(bertmodel,bert_ckpt_file)

import math
def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
    print('create_learning_rate_scheduler')
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

import datetime
log_dir = ".log/movie_reviews/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

total_epoch_count = 10
print('train_x = ')
print(np.array(train_x).shape)
#train_x = (2560,178)
model.fit(x=train_x, y=train_y,
          validation_split=0.1,
          batch_size=24,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=5,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                     tensorboard_callback])
model.save_weights('./movie_reviews.h5', overwrite=True)


_, train_acc = model.evaluate(train_x, train_y)
_, test_acc = model.evaluate(test_x, test_y)

print("train acc", train_acc)
print(" test acc", test_acc)


model = model1
model.load_weights("movie_reviews.h5")

_, train_acc = model.evaluate(train_x, train_y)
_, test_acc = model.evaluate(test_x, test_y)

print("train acc", train_acc)
print(" test acc", test_acc)

pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
pred_tokens    = map(tokenizer.tokenize, pred_sentences)
pred_tokens    = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

res = model.predict(pred_token_ids).argmax(axis=-1)

for text, sentiment in zip(pred_sentences, res):
    print(" text:", text)
    print("  res:", ["negative","positive"][sentiment])