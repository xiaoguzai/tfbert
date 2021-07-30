# gpt2模型简介

gpt2模型与bert模型类似，区别在于gpt2模型之中使用了transformer的decoder部分的内容，而bert模型使用了transformer的encoder部分的内容。
具体gpt2模型对应的结构如下

![image](https://github.com/boss2020/gpt2model/blob/main/data1.png)

![image](https://github.com/boss2020/gpt2model/blob/main/data2.png)

![image](https://github.com/boss2020/gpt2model/blob/main/data3.png)

**这里面要说明注意的几点内容**

1.32个Transformer结构之中的Mask为MultiMaskAttention,这里面是遮盖的attention,与bert的attention不同

首先,bert之中的attention对应的公式为

$softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$

比如这里的Q = (None,None,8,64),$K^{T}$ = (None,None,64,8),则$Q*K^{T}$ = (None,None,8,8)

此时构成了一个相应的矩阵，正好可以使用斜三角矩阵进行遮盖

下面为4*4的遮盖型矩阵

[True False False False

 True True  False False

 True True  True  False

 True True  True  True]

2.32个Transformer结构之中的倒数第二个Dense()之中的激活函数为gelu
**2020 5.20日更新：去除了对应的不同maxlen输入的限制**
```python
self.input_spec = keras.layers.InputSpec(shape=input_shape)
```
放进不同的输入操作为
```python
import tensorflow.keras as keras
import tensorflow as tf
from models import GPT2
maxlen = 128
input_ids = keras.layers.Input(shape=(maxlen,), dtype='int32', name="input_ids")
gpt2model = GPT2(mlm=True)
outputs = gpt2model(input_ids)
input_ids = keras.layers.Input(shape=(24,),dtype='int32',name="input_ids")
outputs = gpt2model(input_ids)
```
