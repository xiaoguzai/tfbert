**安装库的指令**
```python
pip install bertforyou==1.1.9
```
**下载完后配置完预训练文件以及数据集路径即可运行example文件中的example.py以及mlm+bert采样**
# 小白bert使用说明书
一个使用keras复现的bert模型库
之前在看bert源码的时候，发现bert官网上的源码对于新手很不友好，大量的代码和繁杂的英文，都对新手的理解造成了很大的影响，为此本小白制作了一份使用keras复现的bert源代码内容，为了方便新手理解，缩减了所有难以理解的内容，增加了丰富的代码说明和使用指南，并且在原生的bert之上不套用任何bert模型的衍生内容如albert，robert内容等，只为了增加源代码的可读性。

**目前源代码在tensorflow==2.4.0rc0的版本下面测试成功，如果在tensorflow1下面使用相应的源代码，可能出现不兼容的问题，目前的源代码暂时只支持bert-base的模型参数导入。**

bert的整体结构分为预训练和微调两部分，这里呈现的为fine-tune模型的内容介绍

## 预训练文件的下载

预训练文件可以到谷歌的官方地址位置进行下载，

https://github.com/google-research/bert

下载下面的BERT-Base，如果使用中文的话需要下载Bert-Base Chinese

这里我将预训练文件放到了对应的百度云网盘之中分享出来，由于目前只支持base模型，所以只放置了base模型的权重

链接: https://pan.baidu.com/s/16E4mFUmZoBhODEYWfQd6uA  密码: vhfq

## example使用

运行命令python example.py

运行example.py所需的数据集
链接: https://pan.baidu.com/s/1h5JVTePB4g_9jjzr5YrflA  密码: 7u4r

## keras简介

keras入手相对较为简单，只需要掌握三个对应的函数即可

```python
def __init__():
```

这个函数为初始化函数

```python
def build():
```

这个函数类似于前面的初始化函数，区别在于build()函数只有在第一次call()函数调用之前才会调用build()

```python
def call():
```

前向传播函数，类似于pytorch之中的forward()函数，给出了模型具体的操作

## 模型的整体结构

[模型整体结构图片链接](https://blog.csdn.net/znevegiveup1/article/details/115283917)

## Embeddings层的实现

embeddings分为word_embeddings(字向量),position_embeddings(位置向量),segment_embeddings(文本向量，切割每一句话)三个层组成，

word_embeddings为预训练好的(30522,768)矩阵，其中30522对应vocab.txt之中的30522个单词前后缀，每一个词对应768个维度参数。

```python
#build()
self.word_embeddings_layer = keras.layers.Embedding(
    input_dim = self.vocab_size,#30522
    output_dim = self.embedding_size,#768
    mask_zero = self.mask_zero,
    name = "word_embeddings"
)
#call()
word_embeddings = self.word_embeddings_layer(input_ids)
```

position_embedding这里使用的仍然为预训练的参数(512,768)的矩阵，bert模型限制的最大长度为512。这里实现采用的是切割矩阵的方式，将预训练(512,768)的矩阵切分为(maxlen,768)矩阵。

```python
#build()之中的定义
self.position_embeddings_table = self.add_weight(
    name="position_embeddings",
    dtype=K.floatx(),
    shape=[self.max_position_embeddings,self.embedding_size],
    initializer=self.create_initializer()
)
#call()之中的定义
position_embeddings = tf.slice(self.position_embeddings_table,
                               [0,0],
                               [maxlen,-1])
```

segment_embedding使用：如果输入为一个[Input_ids,Segment_ids]构成的list数组，则分别放入word_embedding和segment_embedding之中
如果输入的只有一个Input_ids的tensor内容，则默认使用segment_ids为全零的tensor，然后分别放入word_embedding和segment_embedding层之中
```python
#build()
self.segment_embeddings_layer = keras.layers.Embedding(
    input_dim = self.token_type_vocab_size,
    output_dim = self.embedding_size,
    mask_zero = self.mask_zero,
    name = "segment_embeddings"
)
#call()
segment_ids = None
if isinstance(input_ids,list):
    assert 2 == len(input_ids),"Expecting inputs to be a [input_ids,token_type_ids] list"
    input_ids,segment_ids = input_ids[0],input_ids[1]
if segment_ids == None:
    segment_ids = tf.zeros_like(input_ids)
word_embeddings = self.word_embeddings_layer(input_ids)
segment_embeddings = self.segment_embeddings_layer(segment_ids)
```

实现完之后三个对应的embedding输出相加

```python
results = word_embeddings+tf.reshape(position_embeddings,currentshape)
#tf.reshape(position_embedding,currentshape)将position_embeddings
#维度由(batch_size,128)变为(1,batch_size,128)
results = results+segment_embeddings
```

## LayerNormalization的实现

bert之中的Normalization使用的是LayerNormalization，对应的数学公式为
$$
x = \frac{x-\mu}{\sqrt{(\sigma)^{2}+\epsilon}}
$$
```python
def build(self, input_shape):
    self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
    self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(), trainable=True)
    self.beta  = self.add_weight(name="beta", shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Zeros(), trainable=True)
    super(LayerNormalization,self).build(input_shape)

def call(self, inputs, **kwargs):                               # pragma: no cover
    x = inputs
    if tf.__version__.startswith("2."):
        mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
    else:
        mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
    inv = self.gamma * tf.math.rsqrt(var + self.epsilon)
    res = x * tf.cast(inv, x.dtype) + tf.cast(self.beta - mean * inv, x.dtype)
    #tf.cast()函数执行tensorflow中张量数据类型转换，转换为x.dtype类型
    #这里面使用的res = (inputs-平均)*(1/根号(方差+1e-12))
    #使用的残差的原始函数为(x-平均)*(1/根号(方差+1e-12))
    return res
```

这里使用tf.nn.moments(x,axes=-1,keepdims=True)取出最后一维的768个数值进行归一化操作，然后套用LayerNormalization的对应数学公式求得结果

## Attention实现

关键代码

1.多头注意力机制

```python
query = transpose_for_scores(query,seq_len)
key = transpose_for_scores(key,seq_len)
```

如果最大长度为128的时候，输入的query = (None,128,768)，这里注意力头为12个，将query形状改变为(None,128,12,64)，同理key = (None,128,12,64)

2.调用注意力公式
$$
Attention(Q,K,V) = softmax(\frac{Q*K^{T}}{\sqrt{d_{k}}})*V
$$
对应代码为

```python
attention_scores = tf.matmul(query,key,transpose_b=True)
attention_scores = attention_scores/tf.sqrt(float(self.size_per_head))
attention_probs = tf.nn.softmax(attention_scores)
value = tf.reshape(value,[batch_size,seq_len,
                          self.num_heads,self.size_per_head])
value = tf.transpose(a=value,perm=[0,2,1,3])
context_layer = tf.matmul(attention_probs,value)
```

3.合并多个注意力头

```python
context_layer = tf.reshape(context_layer,output_shape)
#output_shape = (None,128,768),context_layer = (None,128,12,64)
```

## 残差连接实现

**bert之中的Transformer结构只有encoder部分，没有decoder部分，对应的结构图如下：**

[bert之中Transformer结构图片](https://blog.csdn.net/znevegiveup1/article/details/114585302)

为了更好地方便理解，我制作了一个更符号我的代码风格的机构图，对应内容如下：

[代码中Transformer结构图](https://blog.csdn.net/znevegiveup1/article/details/115283985)

Transformer整体实现的结构如下：

```python
def call(self,inputs,**kwargs):
    embedding_output = self.attention(inputs)
    #注意看transformer结构图，这里是经过attention机制之后
    #再进行相应的残差连接
    residual = embedding_output
    embedding_output = self.dense0(embedding_output)
    embedding_output = self.dropout0(embedding_output)
    embedding_output = self.layer_norm0(residual+embedding_output)
    residual = embedding_output
    embedding_output = self.dense(embedding_output)
    #self.dense对应着feed forward层
    embedding_output = self.dense1(embedding_output)
    embedding_output = self.dropout1(embedding_output)
    embedding_output = self.layer_norm1(residual+embedding_output)
    return embedding_output
```
## 权重矩阵的加载

获取权重矩阵和权重矩阵对应的参数内容

```python
bert_params = bert.weights
param_values = keras.backend.batch_get_value(bert.weights)
```

接下来使用一个字典将bert的权重内容对应到params参数内容之中

```python
transformer_dicts = {
'bert/embeddings/position_embeddings/embeddings:0':'bert/embeddings/position_embeddings',
'bert/embeddings/word_embeddings/embeddings:0':'bert/embeddings/word_embeddings',
'bert/embeddings/segment_embeddings/embeddings:0':'bert/embeddings/token_type_embeddings',
'bert/embeddings/layer_normalization/gamma:0':'bert/embeddings/LayerNorm/gamma',
'bert/embeddings/layer_normalization/beta:0':'bert/embeddings/LayerNorm/beta',
}
for layer_ndx in range(bert.num_layers):
print('layer_ndx = ')
print(layer_ndx)
transformer_dicts.update({
    'bert/transformer_%d/attention/query/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/query/kernel'%(layer_ndx),
    #注意中间有冒号，两边要分开进行赋值
    'bert/transformer_%d/attention/query/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/query/bias'%(layer_ndx),
    'bert/transformer_%d/attention/key/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/key/kernel'%(layer_ndx),
    'bert/transformer_%d/attention/key/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/key/bias'%(layer_ndx),
    'bert/transformer_%d/attention/value/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/value/kernel'%(layer_ndx),
    'bert/transformer_%d/attention/value/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/value/bias'%(layer_ndx),

    'bert/transformer_%d/dense0/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/dense/kernel'%(layer_ndx),
    'bert/transformer_%d/dense0/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/dense/bias'%(layer_ndx),
    'bert/transformer_%d/layer_normalization_0/gamma:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/LayerNorm/gamma'%(layer_ndx),
    'bert/transformer_%d/layer_normalization_0/beta:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/LayerNorm/beta'%(layer_ndx),

    'bert/transformer_%d/dense/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/intermediate/dense/kernel'%(layer_ndx),
    'bert/transformer_%d/dense/bias:0'%(layer_ndx):'bert/encoder/layer_%d/intermediate/dense/bias'%(layer_ndx),

    'bert/transformer_%d/dense1/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/output/dense/kernel'%(layer_ndx),
    'bert/transformer_%d/dense1/bias:0'%(layer_ndx):'bert/encoder/layer_%d/output/dense/bias'%(layer_ndx),
    'bert/transformer_%d/layer_normalization_1/gamma:0'%(layer_ndx):'bert/encoder/layer_%d/output/LayerNorm/gamma'%(layer_ndx),
    'bert/transformer_%d/layer_normalization_1/beta:0'%(layer_ndx):'bert/encoder/layer_%d/output/LayerNorm/beta'%(layer_ndx)

})
```

遍历现在的权重参数，以元组形式放入对应的set()之中

```python
for ndx,(param_value,param) in enumerate(zip(param_values,bert_params)):
#将对应的权重值放入相应的set()之中
#param为对应的权重参数，而ckpt_value为预训练的参数值
	weight_value_tuples.append((param,ckpt_value))
```

重置weight_value_tuples的内容

```python
keras.backend.batch_set_value(weight_value_tuples)
```
## 注意事项
这里面的有一些参数尚未使用上去
bert/pooler/dense/bias
bert/pooler/dense/kernel
cls/predictions/output_bias
cls/predictions/transform/LayerNorm/beta
cls/predictions/transform/LayerNorm/gamma
cls/predictions/transform/dense/bias
cls/predictions/transform/dense/kernel
cls/seq_relationship/output_bias
cls/seq_relationship/output_weights
这是因为这里的参数是用于其他一些任务的，比如seq_relationship用于序列的关系，
predictions权重用于句子的预测，pooler池化层也是用于nsp等任务的，分类任务这些
权重内容暂时使用不上，所以分类任务暂时不需要这些权重的内容

=====================================================================================================
## 更新内容
2021年3月31号：更新了loader之中的权重定义，以及对应的segment_ids有输入和没有输入的不同的实现
**2021年4月2号：更新了intermediate layer之中的定义**
```
self.dense = keras.layers.Dense(units = self.intermediate_size,
				kernel_initializer = self.create_initializer(),
				activation = self.get_activation('gelu'),
				name = "dense")
```
**intermediate层之中加入了激活函数，发现对于深度学习收敛过程帮助很大**
使用的调试小技巧类似于
```python
inputs = [1,2,3,4,5]
outputs = model(inputs)
```
观察模型对应的输出情况
**2021年4月2日：更新了example.py的内容，解决了输入的keras.Input中的batch_size和max_len与最终输入数据中的batch_size和max_len维度不同造成的结果无法收敛的情形**
即之前的
```python
input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
```
**上面Input之中的max_seq_len = 128**
和最终输入的
```python
model.fit(x=train_x, y=train_y,
          validation_split=0.1,
          batch_size=batch_size,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=5,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                     #EarlyStopping用于提前停止训练的callbacks,可以达到当训练集上的loss
                     #不再减小(即减小的程度小于某个阀值)的时候停止继续训练
                     #patience:能够容忍多少个epoch内都没有improvement
                     tensorboard_callback])
```
中的np.array(x).shape中的max_len维度不同造成的不能很好地收敛的情况
**上面Input之中的max_seq_len = 178**
维度不同的时候对应的结果不能够很好的收敛，
而当调整维度相同之后对应的结果：
```
Epoch 1/10

Epoch 00001: LearningRateScheduler reducing learning rate to 5e-06.
WARNING:tensorflow:Gradients do not exist for variables ['bert/embeddings/segment_embeddings/embeddings:0'] when minimizing the loss.
WARNING:tensorflow:Gradients do not exist for variables ['bert/embeddings/segment_embeddings/embeddings:0'] when minimizing the loss.
48/48 [==============================] - 28s 493ms/step - loss: 0.7870 - acc: 0.5460 - val_loss: 0.5164 - val_acc: 0.8281
Epoch 2/10

Epoch 00002: LearningRateScheduler reducing learning rate to 1e-05.
48/48 [==============================] - 22s 462ms/step - loss: 0.4543 - acc: 0.7780 - val_loss: 0.3099 - val_acc: 0.8516
Epoch 3/10

Epoch 00003: LearningRateScheduler reducing learning rate to 1.5000000000000002e-05.
48/48 [==============================] - 23s 471ms/step - loss: 0.2313 - acc: 0.9099 - val_loss: 0.3257 - val_acc: 0.8828
Epoch 4/10

Epoch 00004: LearningRateScheduler reducing learning rate to 2e-05.
48/48 [==============================] - 23s 473ms/step - loss: 0.1546 - acc: 0.9376 - val_loss: 0.3409 - val_acc: 0.9023
Epoch 5/10

Epoch 00005: LearningRateScheduler reducing learning rate to 2.5e-05.
48/48 [==============================] - 23s 478ms/step - loss: 0.0807 - acc: 0.9756 - val_loss: 0.4122 - val_acc: 0.9062
Epoch 6/10

Epoch 00006: LearningRateScheduler reducing learning rate to 3.0000000000000004e-05.
48/48 [==============================] - 23s 475ms/step - loss: 0.0261 - acc: 0.9892 - val_loss: 0.5326 - val_acc: 0.8906
Epoch 7/10

Epoch 00007: LearningRateScheduler reducing learning rate to 3.5000000000000004e-05.
48/48 [==============================] - 23s 479ms/step - loss: 0.0201 - acc: 0.9917 - val_loss: 0.5701 - val_acc: 0.8828
Epoch 8/10

Epoch 00008: LearningRateScheduler reducing learning rate to 4e-05.
48/48 [==============================] - 23s 477ms/step - loss: 0.0541 - acc: 0.9814 - val_loss: 0.5465 - val_acc: 0.8945
Epoch 9/10

Epoch 00009: LearningRateScheduler reducing learning rate to 4.5e-05.
48/48 [==============================] - 23s 478ms/step - loss: 0.0245 - acc: 0.9922 - val_loss: 0.5279 - val_acc: 0.8945
Epoch 10/10

Epoch 00010: LearningRateScheduler reducing learning rate to 5e-05.
48/48 [==============================] - 23s 478ms/step - loss: 0.0295 - acc: 0.9918 - val_loss: 0.7450 - val_acc: 0.8789
```
可以看出调整max_seq_len维度相同之后模型能够很好地收敛
**另外一个就是如果你是从config之中读取对应的参数的时候，需要注意如果加入batch_size的话，你的batch_size要与最终输入的batch_size数值一致**
```python
with open(bert_config_file) as f:
    config = json.load(f)
config['batch_size'] = 48
print(config)
```
与模型训练中的
```python
model.fit(x=train_x, y=train_y,
          validation_split=0.1,
          batch_size=batch_size,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=5,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                     #EarlyStopping用于提前停止训练的callbacks,可以达到当训练集上的loss
                     #不再减小(即减小的程度小于某个阀值)的时候停止继续训练
                     #patience:能够容忍多少个epoch内都没有improvement
                     tensorboard_callback])
```
内容保持一致
**2021年4月3日：加入了bert之中采用mlm进行采样的操作过程**

具体添加的网络层采样代码如下
```python
if self.with_mlm:
    self.mlm_dense0 = keras.layers.Dense(units = self.embedding_size,
				kernel_initializer = self.create_initializer(),
				name = "mlm_dense0")
    self.mlm_norm = LayerNormalization()
    self.mlm_norm.name = 'mlm_norm'
    self.mlm_dense1 = keras.layers.Dense(units = self.vocab_size,
				kernel_initializer = self.create_initializer(),
				name = "mlm_dense1")
```
可以看出这里先添加了一个dense层，再添加了一个归一化层，最后再添加了一个dense层内容
```python
if self.with_mlm:
    outputs = self.mlm_dense0(outputs)
    outputs = self.mlm_norm(outputs)
    outputs = self.mlm_dense1(outputs)
```
对应添加的权重加载过程如下：
```python
'bert/mlm_dense0/kernel:0':'cls/predictions/transform/dense/kernel',
'bert/mlm_dense0/bias:0':'cls/predictions/transform/dense/bias',
'bert/mlm_norm/gamma:0':'cls/predictions/transform/LayerNorm/gamma',
'bert/mlm_norm/beta:0':'cls/predictions/transform/LayerNorm/beta',
'bert/mlm_dense1/kernel:0':'bert/embeddings/word_embeddings',
'bert/mlm_dense1/bias:0':'cls/predictions/output_bias'
```
第一个dense层使用了'cls/predictions/transform/dense/kernel'以及'cls/predictions/transform/dense/bias'的权重
(None,128,768)，
第二个layernormalization使用了'cls/predictions/transform/LayerNorm/gamma'以及'cls/predictions/transform/LayerNorm/beta'的权重
(None,128,768)，
第三个dense层使用了word_embedding权重的转置(21128,768)->(768,21128)
(None,128,768)乘(768,21128) = (None,128,21128)
最后取出第1句第i个单词的每一个单词表中的概率(0,i)->(21128,),然后从中随机抽样出单词的概率

**2021.4.30：更新了bert采样的过程，注意mlm中的对应的dense层有一定的微调**

```python
if self.with_mlm:
    self.mlm_dense0 = keras.layers.Dense(units = self.embedding_size,
                                         kernel_initializer = self.create_initializer(),
                                         activation = self.get_activation('gelu'),
                                         name = "mlm_dense0")
```

这里加入了一个对应的'gelu'激活函数的内容

更新了使用掩码遮盖之后反向预测对应词语的过程，对应代码内容在mlm预测遮盖掩码过程.py中

**2021.5.27：更新了bert之中position_embedding内容的实现。**

原先bert之中的position_embedding是通过截取权重矩阵实现的

原先部分定义对应的权重矩阵：

build()函数部分的操作

```python
self.position_embeddings_table = self.add_weight(
    name="position_embeddings/embeddings",
    dtype=K.floatx(),
    shape=[self.max_position_embeddings,self.embedding_size],
    initializer=self.create_initializer()
)
```

在操作的过程之中截取相应的权重

call()函数部分的操作

```python
position_embeddings = self.position_embeddings_table[None,:seq_len]
...
results = word_embeddings+position_embeddings
```

此时在训练的时候维度报错如下，看不出来是哪一句代码发生了错误

```
InvalidArgumentError: 2 root error(s) found.
  (0) Invalid argument:  required broadcastable shapes at loc(unknown)
	 [[node model/bert/embeddings/add (defined at /home/xiaoguzai/代码/tfbert-main/models.py:332) ]]
  (1) Invalid argument:  required broadcastable shapes at loc(unknown)
	 [[node model/bert/embeddings/add (defined at /home/xiaoguzai/代码/tfbert-main/models.py:332) ]]
	 [[gradient_tape/model/bert/embeddings/word_embeddings/embedding_lookup/Reshape/_644]]
0 successful operations.
0 derived errors ignored. [Op:__inference_train_function_29474]
```

**但是根据required broadcastable shapes at loc(unknown)以及报错的对应代码语句**

```python
position_embeddings = self.position_embeddings_table[None,:seq_len]
```

我们知道是这里的position_embeddings的形状可能发生了变化

此时这里我们修改一下，使用相应的embedding网络层实现position_embeddings的部分：

build()函数部分的操作：

```python
self.position_embeddings_layer = keras.layers.Embedding(
	input_dim = self.max_position_embeddings,
    output_dim = self.embedding_size,
    mask_zero = self.mask_zero,
    name = "position_embeddings"
)
```

定义相应的Embedding网络层

在call()函数中的相应的操作：

```python
position_embeddings = self.position_embeddings_layer(position_ids)
```

这时候通过报错就能够看出来了

```
InvalidArgumentError: 2 root error(s) found.
  (0) Invalid argument:  indices[0,558] = 558 is not in [0, 512)
	 [[node model/bert/embeddings/position_embeddings/embedding_lookup (defined at /home/xiaoguzai/代码/tfbert-main/models.py:300) ]]
	 [[model/bert/embeddings/word_embeddings/embedding_lookup/_26]]
  (1) Invalid argument:  indices[0,558] = 558 is not in [0, 512)
	 [[node model/bert/embeddings/position_embeddings/embedding_lookup (defined at /home/xiaoguzai/代码/tfbert-main/models.py:300) ]]
0 successful operations.
0 derived errors ignored. [Op:__inference_train_function_29522]
```

也就是说，558超出了最大的position_embedding的512的数值，所以这里会进行相应的报错

**2020.6.17修改中间的gelu激活函数**

原先激活函数的定义为

```python
def gelu(self,x):
    """
        Gelu activation from arXiv:1606.08415.
        """
    cdf = 0.5 * (1.0 + tf.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    ))
    return x * cdf
```



中间的gelu激活函数可以修改成为另外一种实现方式

```python
def gelu(self,x):
    """
        Gelu activation from arXiv:1606.08415.
        """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))
    return x * cdf
```