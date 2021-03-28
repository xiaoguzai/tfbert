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

见example.py文件

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

