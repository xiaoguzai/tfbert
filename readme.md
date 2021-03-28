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

segment_embedding使用的是bert预训练的token_embedding，主要用于分割句子，第一句的标志为0，第二句的标志为1，第三句标志为0这样依次分割下去。目前这里的实现还是简易版本，标志为全零标志，日后还会进行进一步更新。

```python
#build()
self.segment_embeddings_layer = keras.layers.Embedding(
    input_dim = self.token_type_vocab_size,
    output_dim = self.embedding_size,
    mask_zero = self.mask_zero,
    name = "segment_embeddings"
)
#call()
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
       'bert/embeddings/position_embeddings:0':'bert/embeddings/position_embeddings',
       'bert/embeddings/word_embeddings/embeddings:0':'bert/embeddings/word_embeddings',
       'bert/embeddings/segment_embeddings/embeddings:0':'bert/embeddings/token_type_embeddings',
       'bert/embeddings/layer_normalization/gamma:0':'bert/embeddings/LayerNorm/gamma',
       'bert/embeddings/layer_normalization/beta:0':'bert/embeddings/LayerNorm/beta',
        #1
       'bert/transformer/attention_layer/query/kernel:0':'bert/encoder/layer_0/attention/self/query/kernel',
       'bert/transformer/attention_layer/query/bias:0':'bert/encoder/layer_0/attention/self/query/bias',
       'bert/transformer/attention_layer/key/kernel:0':'bert/encoder/layer_0/attention/self/key/kernel',
       'bert/transformer/attention_layer/key/bias:0':'bert/encoder/layer_0/attention/self/key/bias',
       'bert/transformer/attention_layer/value/kernel:0':'bert/encoder/layer_0/attention/self/value/kernel',
       'bert/transformer/attention_layer/value/bias:0':'bert/encoder/layer_0/attention/self/value/bias',
        #...
       'bert/transformer/dense0/kernel:0':'bert/encoder/layer_0/attention/output/dense/kernel',
       'bert/transformer/dense0/bias:0':'bert/encoder/layer_0/attention/output/dense/bias',
       'bert/transformer/layer_normalization_1/gamma:0':'bert/encoder/layer_0/attention/output/LayerNorm/gamma',
       'bert/transformer/layer_normalization_1/beta:0':'bert/encoder/layer_0/attention/output/LayerNorm/beta',
        #...
       'bert/transformer/dense/kernel:0':'bert/encoder/layer_0/intermediate/dense/kernel',#未使用
       'bert/transformer/dense/bias:0':'bert/encoder/layer_0/intermediate/dense/bias',#未使用
        #...
       'bert/transformer/dense1/kernel:0':'bert/encoder/layer_0/output/dense/kernel',#未使用
       'bert/transformer/dense1/bias:0':'bert/encoder/layer_0/output/dense/bias',
       'bert/transformer/layer_normalization_2/gamma:0':'bert/encoder/layer_0/output/LayerNorm/gamma',
       'bert/transformer/layer_normalization_2/beta:0':'bert/encoder/layer_0/output/LayerNorm/beta',
        #...
        #2
       'bert/transformer_1/attention_layer_1/query/kernel:0':'bert/encoder/layer_1/attention/self/query/kernel', 
       'bert/transformer_1/attention_layer_1/query/bias:0':'bert/encoder/layer_1/attention/self/query/bias',
       'bert/transformer_1/attention_layer_1/key/kernel:0':'bert/encoder/layer_1/attention/self/key/kernel',
       'bert/transformer_1/attention_layer_1/key/bias:0':'bert/encoder/layer_1/attention/self/key/bias',
       'bert/transformer_1/attention_layer_1/value/kernel:0':'bert/encoder/layer_1/attention/self/value/kernel',
       'bert/transformer_1/attention_layer_1/value/bias:0':'bert/encoder/layer_1/attention/self/value/bias',
        #...
       'bert/transformer_1/dense0/kernel:0':'bert/encoder/layer_1/attention/output/dense/kernel',
       'bert/transformer_1/dense0/bias:0':'bert/encoder/layer_1/attention/output/dense/bias',
       'bert/transformer_1/layer_normalization_3/gamma:0':'bert/encoder/layer_1/attention/output/LayerNorm/gamma',
       'bert/transformer_1/layer_normalization_3/beta:0':'bert/encoder/layer_1/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_1/dense/kernel:0':'bert/encoder/layer_1/intermediate/dense/kernel',#未使用
       'bert/transformer_1/dense/bias:0':'bert/encoder/layer_1/intermediate/dense/bias',#未使用
        #...
       'bert/transformer_1/dense1/kernel:0':'bert/encoder/layer_1/output/dense/kernel',#未使用
       'bert/transformer_1/dense1/bias:0':'bert/encoder/layer_1/output/dense/bias',
       'bert/transformer_1/layer_normalization_4/gamma:0':'bert/encoder/layer_1/output/LayerNorm/gamma',
       'bert/transformer_1/layer_normalization_4/beta:0':'bert/encoder/layer_1/output/LayerNorm/beta', 
        #...
        #3
       'bert/transformer_2/attention_layer_2/query/kernel:0':'bert/encoder/layer_2/attention/self/query/kernel',
       'bert/transformer_2/attention_layer_2/query/bias:0':'bert/encoder/layer_2/attention/self/query/bias',
       'bert/transformer_2/attention_layer_2/key/kernel:0':'bert/encoder/layer_2/attention/self/key/kernel',
       'bert/transformer_2/attention_layer_2/key/bias:0':'bert/encoder/layer_2/attention/self/key/bias',
       'bert/transformer_2/attention_layer_2/value/kernel:0':'bert/encoder/layer_2/attention/self/value/kernel',
       'bert/transformer_2/attention_layer_2/value/bias:0':'bert/encoder/layer_2/attention/self/value/bias',
        #...
       'bert/transformer_2/dense0/kernel:0':'bert/encoder/layer_2/attention/output/dense/kernel',
       'bert/transformer_2/dense0/bias:0':'bert/encoder/layer_2/attention/output/dense/bias',
       'bert/transformer_2/layer_normalization_5/gamma:0':'bert/encoder/layer_2/attention/output/LayerNorm/gamma',
       'bert/transformer_2/layer_normalization_5/beta:0':'bert/encoder/layer_2/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_2/dense/kernel:0':'bert/encoder/layer_2/intermediate/dense/kernel',
       'bert/transformer_2/dense/bias:0':'bert/encoder/layer_2/intermediate/dense/bias',
        #...
       'bert/transformer_2/dense1/kernel:0':'bert/encoder/layer_2/output/dense/kernel',
       'bert/transformer_2/dense1/bias:0':'bert/encoder/layer_2/output/dense/bias',
       'bert/transformer_2/layer_normalization_6/gamma:0':'bert/encoder/layer_2/output/LayerNorm/gamma',
       'bert/transformer_2/layer_normalization_6/beta:0':'bert/encoder/layer_2/output/LayerNorm/beta',
        #...
        #4
       'bert/transformer_3/attention_layer_3/query/kernel:0':'bert/encoder/layer_3/attention/self/query/kernel',
       'bert/transformer_3/attention_layer_3/query/bias:0':'bert/encoder/layer_3/attention/self/query/bias',
       'bert/transformer_3/attention_layer_3/key/kernel:0':'bert/encoder/layer_3/attention/self/key/kernel',
       'bert/transformer_3/attention_layer_3/key/bias:0':'bert/encoder/layer_3/attention/self/key/bias',
       'bert/transformer_3/attention_layer_3/value/kernel:0':'bert/encoder/layer_3/attention/self/value/kernel',
       'bert/transformer_3/attention_layer_3/value/bias:0':'bert/encoder/layer_3/attention/self/value/bias',
        #...
       'bert/transformer_3/dense0/kernel:0':'bert/encoder/layer_3/attention/output/dense/kernel',
       'bert/transformer_3/dense0/bias:0':'bert/encoder/layer_3/attention/output/dense/bias',
       'bert/transformer_3/layer_normalization_7/gamma:0':'bert/encoder/layer_3/attention/output/LayerNorm/gamma',
       'bert/transformer_3/layer_normalization_7/beta:0':'bert/encoder/layer_3/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_3/dense/kernel:0':'bert/encoder/layer_3/intermediate/dense/kernel',
       'bert/transformer_3/dense/bias:0':'bert/encoder/layer_3/intermediate/dense/bias',
        #...
       'bert/transformer_3/dense1/kernel:0':'bert/encoder/layer_3/output/dense/kernel',
       'bert/transformer_3/dense1/bias:0':'bert/encoder/layer_3/output/dense/bias',
       'bert/transformer_3/layer_normalization_8/gamma:0':'bert/encoder/layer_3/output/LayerNorm/gamma',
       'bert/transformer_3/layer_normalization_8/beta:0':'bert/encoder/layer_3/output/LayerNorm/beta',
        #...
        #5
       'bert/transformer_4/attention_layer_4/query/kernel:0':'bert/encoder/layer_4/attention/self/query/kernel',
       'bert/transformer_4/attention_layer_4/query/bias:0':'bert/encoder/layer_4/attention/self/query/bias',
       'bert/transformer_4/attention_layer_4/key/kernel:0':'bert/encoder/layer_4/attention/self/key/kernel',
       'bert/transformer_4/attention_layer_4/key/bias:0':'bert/encoder/layer_4/attention/self/key/bias',
       'bert/transformer_4/attention_layer_4/value/kernel:0':'bert/encoder/layer_4/attention/self/value/kernel',
       'bert/transformer_4/attention_layer_4/value/bias:0':'bert/encoder/layer_4/attention/self/value/bias',
        #...
       'bert/transformer_4/dense0/kernel:0':'bert/encoder/layer_4/attention/output/dense/kernel',
       'bert/transformer_4/dense0/bias:0':'bert/encoder/layer_4/attention/output/dense/bias',
       'bert/transformer_4/layer_normalization_9/gamma:0':'bert/encoder/layer_4/attention/output/LayerNorm/gamma',
       'bert/transformer_4/layer_normalization_9/beta:0':'bert/encoder/layer_4/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_4/dense/kernel:0':'bert/encoder/layer_4/intermediate/dense/kernel',
       'bert/transformer_4/dense/bias:0':'bert/encoder/layer_4/intermediate/dense/bias',
        #...
       'bert/transformer_4/dense1/kernel:0':'bert/encoder/layer_4/output/dense/kernel',
       'bert/transformer_4/dense1/bias:0':'bert/encoder/layer_4/output/dense/bias',
       'bert/transformer_4/layer_normalization_10/gamma:0':'bert/encoder/layer_4/output/LayerNorm/gamma',
       'bert/transformer_4/layer_normalization_10/beta:0':'bert/encoder/layer_4/output/LayerNorm/beta',
        #...
        #6
       'bert/transformer_5/attention_layer_5/query/kernel:0':'bert/encoder/layer_5/attention/self/query/kernel',
       'bert/transformer_5/attention_layer_5/query/bias:0':'bert/encoder/layer_5/attention/self/query/bias',
       'bert/transformer_5/attention_layer_5/key/kernel:0':'bert/encoder/layer_5/attention/self/key/kernel',
       'bert/transformer_5/attention_layer_5/key/bias:0':'bert/encoder/layer_5/attention/self/key/bias',
       'bert/transformer_5/attention_layer_5/value/kernel:0':'bert/encoder/layer_5/attention/self/value/kernel',
       'bert/transformer_5/attention_layer_5/value/bias:0':'bert/encoder/layer_5/attention/self/value/bias',
        #...
       'bert/transformer_5/dense0/kernel:0':'bert/encoder/layer_5/attention/output/dense/kernel',
       'bert/transformer_5/dense0/bias:0':'bert/encoder/layer_5/attention/output/dense/bias',
       'bert/transformer_5/layer_normalization_11/gamma:0':'bert/encoder/layer_5/attention/output/LayerNorm/gamma',
       'bert/transformer_5/layer_normalization_11/beta:0':'bert/encoder/layer_5/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_5/dense/kernel:0':'bert/encoder/layer_5/intermediate/dense/kernel',
       'bert/transformer_5/dense/bias:0':'bert/encoder/layer_5/intermediate/dense/bias',
        #...
       'bert/transformer_5/dense1/kernel:0':'bert/encoder/layer_5/output/dense/kernel',
       'bert/transformer_5/dense1/bias:0':'bert/encoder/layer_5/output/dense/bias',
       'bert/transformer_5/layer_normalization_12/gamma:0':'bert/encoder/layer_5/output/LayerNorm/gamma',
       'bert/transformer_5/layer_normalization_12/beta:0':'bert/encoder/layer_5/output/LayerNorm/beta',
        #...
        #7
       'bert/transformer_6/attention_layer_6/query/kernel:0':'bert/encoder/layer_6/attention/self/query/kernel',
       'bert/transformer_6/attention_layer_6/query/bias:0':'bert/encoder/layer_6/attention/self/query/bias',
       'bert/transformer_6/attention_layer_6/key/kernel:0':'bert/encoder/layer_6/attention/self/key/kernel',
       'bert/transformer_6/attention_layer_6/key/bias:0':'bert/encoder/layer_6/attention/self/key/bias',
       'bert/transformer_6/attention_layer_6/value/kernel:0':'bert/encoder/layer_6/attention/self/value/kernel',
       'bert/transformer_6/attention_layer_6/value/bias:0':'bert/encoder/layer_6/attention/self/value/bias',
        #...
       'bert/transformer_6/dense0/kernel:0':'bert/encoder/layer_6/attention/output/dense/kernel',
       'bert/transformer_6/dense0/bias:0':'bert/encoder/layer_6/attention/output/dense/bias',
       'bert/transformer_6/layer_normalization_13/gamma:0':'bert/encoder/layer_6/attention/output/LayerNorm/gamma',
       'bert/transformer_6/layer_normalization_13/beta:0':'bert/encoder/layer_6/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_6/dense/kernel:0':'bert/encoder/layer_6/intermediate/dense/kernel',
       'bert/transformer_6/dense/bias:0':'bert/encoder/layer_6/intermediate/dense/bias',
        #...
       'bert/transformer_6/dense1/kernel:0':'bert/encoder/layer_6/output/dense/kernel',
       'bert/transformer_6/dense1/bias:0':'bert/encoder/layer_6/output/dense/bias',
       'bert/transformer_6/layer_normalization_14/gamma:0':'bert/encoder/layer_6/output/LayerNorm/gamma',
       'bert/transformer_6/layer_normalization_14/beta:0':'bert/encoder/layer_6/output/LayerNorm/beta',
        #...
        #8
       'bert/transformer_7/attention_layer_7/query/kernel:0':'bert/encoder/layer_7/attention/self/query/kernel',
       'bert/transformer_7/attention_layer_7/query/bias:0':'bert/encoder/layer_7/attention/self/query/bias',
       'bert/transformer_7/attention_layer_7/key/kernel:0':'bert/encoder/layer_7/attention/self/key/kernel',
       'bert/transformer_7/attention_layer_7/key/bias:0':'bert/encoder/layer_7/attention/self/key/bias',
       'bert/transformer_7/attention_layer_7/value/kernel:0':'bert/encoder/layer_7/attention/self/value/kernel',
       'bert/transformer_7/attention_layer_7/value/bias:0':'bert/encoder/layer_7/attention/self/value/bias',
        #...
       'bert/transformer_7/dense0/kernel:0':'bert/encoder/layer_7/attention/output/dense/kernel',
       'bert/transformer_7/dense0/bias:0':'bert/encoder/layer_7/attention/output/dense/bias',
       'bert/transformer_7/layer_normalization_15/gamma:0':'bert/encoder/layer_7/attention/output/LayerNorm/gamma',
       'bert/transformer_7/layer_normalization_15/beta:0':'bert/encoder/layer_7/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_7/dense/kernel:0':'bert/encoder/layer_7/intermediate/dense/kernel',
       'bert/transformer_7/dense/bias:0':'bert/encoder/layer_7/intermediate/dense/bias',
        #...
       'bert/transformer_7/dense1/kernel:0':'bert/encoder/layer_7/output/dense/kernel',
       'bert/transformer_7/dense1/bias:0':'bert/encoder/layer_7/output/dense/bias',
       'bert/transformer_7/layer_normalization_16/gamma:0':'bert/encoder/layer_7/output/LayerNorm/gamma',
       'bert/transformer_7/layer_normalization_16/beta:0':'bert/encoder/layer_7/output/LayerNorm/beta',
        #...
        #9
       'bert/transformer_8/attention_layer_8/query/kernel:0':'bert/encoder/layer_8/attention/self/query/kernel',
       'bert/transformer_8/attention_layer_8/query/bias:0':'bert/encoder/layer_8/attention/self/query/bias',
       'bert/transformer_8/attention_layer_8/key/kernel:0':'bert/encoder/layer_8/attention/self/key/kernel',
       'bert/transformer_8/attention_layer_8/key/bias:0':'bert/encoder/layer_8/attention/self/key/bias',
       'bert/transformer_8/attention_layer_8/value/kernel:0':'bert/encoder/layer_8/attention/self/value/kernel',
       'bert/transformer_8/attention_layer_8/value/bias:0':'bert/encoder/layer_8/attention/self/value/bias',
        #...
       'bert/transformer_8/dense0/kernel:0':'bert/encoder/layer_8/attention/output/dense/kernel',
       'bert/transformer_8/dense0/bias:0':'bert/encoder/layer_8/attention/output/dense/bias',
       'bert/transformer_8/layer_normalization_17/gamma:0':'bert/encoder/layer_8/attention/output/LayerNorm/gamma',
       'bert/transformer_8/layer_normalization_17/beta:0':'bert/encoder/layer_8/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_8/dense/kernel:0':'bert/encoder/layer_8/intermediate/dense/kernel',
       'bert/transformer_8/dense/bias:0':'bert/encoder/layer_8/intermediate/dense/bias',
        #...
       'bert/transformer_8/dense1/kernel:0':'bert/encoder/layer_8/output/dense/kernel',
       'bert/transformer_8/dense1/bias:0':'bert/encoder/layer_8/output/dense/bias',
       'bert/transformer_8/layer_normalization_18/gamma:0':'bert/encoder/layer_8/output/LayerNorm/gamma',
       'bert/transformer_8/layer_normalization_18/beta:0':'bert/encoder/layer_8/output/LayerNorm/beta',
        #...
        #10
       'bert/transformer_9/attention_layer_9/query/kernel:0':'bert/encoder/layer_9/attention/self/query/kernel',
       'bert/transformer_9/attention_layer_9/query/bias:0':'bert/encoder/layer_9/attention/self/query/bias',
       'bert/transformer_9/attention_layer_9/key/kernel:0':'bert/encoder/layer_9/attention/self/key/kernel',
       'bert/transformer_9/attention_layer_9/key/bias:0':'bert/encoder/layer_9/attention/self/key/bias',
       'bert/transformer_9/attention_layer_9/value/kernel:0':'bert/encoder/layer_9/attention/self/value/kernel',
       'bert/transformer_9/attention_layer_9/value/bias:0':'bert/encoder/layer_9/attention/self/value/bias',
        #...
       'bert/transformer_9/dense0/kernel:0':'bert/encoder/layer_9/attention/output/dense/kernel',
       'bert/transformer_9/dense0/bias:0':'bert/encoder/layer_9/attention/output/dense/bias',
       'bert/transformer_9/layer_normalization_19/gamma:0':'bert/encoder/layer_9/attention/output/LayerNorm/gamma',
       'bert/transformer_9/layer_normalization_19/beta:0':'bert/encoder/layer_9/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_9/dense/kernel:0':'bert/encoder/layer_9/intermediate/dense/kernel',
       'bert/transformer_9/dense/bias:0':'bert/encoder/layer_9/intermediate/dense/bias',
        #...
       'bert/transformer_9/dense1/kernel:0':'bert/encoder/layer_9/output/dense/kernel',
       'bert/transformer_9/dense1/bias:0':'bert/encoder/layer_9/output/dense/bias',
       'bert/transformer_9/layer_normalization_20/gamma:0':'bert/encoder/layer_9/output/LayerNorm/gamma',
       'bert/transformer_9/layer_normalization_20/beta:0':'bert/encoder/layer_9/output/LayerNorm/beta',
        #11
       'bert/transformer_10/attention_layer_10/query/kernel:0':'bert/encoder/layer_10/attention/self/query/kernel',
       'bert/transformer_10/attention_layer_10/query/bias:0':'bert/encoder/layer_10/attention/self/query/bias',
       'bert/transformer_10/attention_layer_10/key/kernel:0':'bert/encoder/layer_10/attention/self/key/kernel',
       'bert/transformer_10/attention_layer_10/key/bias:0':'bert/encoder/layer_10/attention/self/key/bias',
       'bert/transformer_10/attention_layer_10/value/kernel:0':'bert/encoder/layer_10/attention/self/value/kernel',
       'bert/transformer_10/attention_layer_10/value/bias:0':'bert/encoder/layer_10/attention/self/value/bias',
        #...
       'bert/transformer_10/dense0/kernel:0':'bert/encoder/layer_10/attention/output/dense/kernel',
       'bert/transformer_10/dense0/bias:0':'bert/encoder/layer_10/attention/output/dense/bias',
       'bert/transformer_10/layer_normalization_21/gamma:0':'bert/encoder/layer_10/attention/output/LayerNorm/gamma',
       'bert/transformer_10/layer_normalization_21/beta:0':'bert/encoder/layer_10/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_10/dense/kernel:0':'bert/encoder/layer_10/intermediate/dense/kernel',
       'bert/transformer_10/dense/bias:0':'bert/encoder/layer_10/intermediate/dense/bias',
        #...
       'bert/transformer_10/dense1/kernel:0':'bert/encoder/layer_10/output/dense/kernel',
       'bert/transformer_10/dense1/bias:0':'bert/encoder/layer_10/output/dense/bias',
       'bert/transformer_10/layer_normalization_22/gamma:0':'bert/encoder/layer_10/output/LayerNorm/gamma',
       'bert/transformer_10/layer_normalization_22/beta:0':'bert/encoder/layer_10/output/LayerNorm/beta', 
        #12
       'bert/transformer_11/attention_layer_11/query/kernel:0':'bert/encoder/layer_11/attention/self/query/kernel',
       'bert/transformer_11/attention_layer_11/query/bias:0':'bert/encoder/layer_11/attention/self/query/bias',
       'bert/transformer_11/attention_layer_11/key/kernel:0':'bert/encoder/layer_11/attention/self/key/kernel',
       'bert/transformer_11/attention_layer_11/key/bias:0':'bert/encoder/layer_11/attention/self/key/bias',
       'bert/transformer_11/attention_layer_11/value/kernel:0':'bert/encoder/layer_11/attention/self/value/kernel',
       'bert/transformer_11/attention_layer_11/value/bias:0':'bert/encoder/layer_11/attention/self/value/bias',
        #...
       'bert/transformer_11/dense0/kernel:0':'bert/encoder/layer_11/attention/output/dense/kernel',
       'bert/transformer_11/dense0/bias:0':'bert/encoder/layer_11/attention/output/dense/bias',
       'bert/transformer_11/layer_normalization_23/gamma:0':'bert/encoder/layer_11/attention/output/LayerNorm/gamma',
       'bert/transformer_11/layer_normalization_23/beta:0':'bert/encoder/layer_11/attention/output/LayerNorm/beta',
        #...
       'bert/transformer_11/dense/kernel:0':'bert/encoder/layer_11/intermediate/dense/kernel',
       'bert/transformer_11/dense/bias:0':'bert/encoder/layer_11/intermediate/dense/bias',
        #...
       'bert/transformer_11/dense1/kernel:0':'bert/encoder/layer_11/output/dense/kernel',
       'bert/transformer_11/dense1/bias:0':'bert/encoder/layer_11/output/dense/bias',
       'bert/transformer_11/layer_normalization_24/gamma:0':'bert/encoder/layer_11/output/LayerNorm/gamma',
       'bert/transformer_11/layer_normalization_24/beta:0':'bert/encoder/layer_11/output/LayerNorm/beta'
    }
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
