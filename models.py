import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import os
import numpy as np
#import tensorflow.keras.layers.Layer as Layer
#No module named 'tensorflow.keras.layers.Layer'
#之前的写法是import tensorflow.keras.layers.Layer as Layer报错
class  Bert(tf.keras.layers.Layer):
    name = 'bert'
    def __init__(self,
                maxlen = 128,#0
                initializer_range=0.02,#1
                max_position_embeddings=512,#2
                embedding_size=768,#4
                project_embeddings_with_bias=True,#5
                vocab_size=30522,#6
                use_token_type=True,#7
                use_position_embeddings=True,#8
                token_type_vocab_size=2,#9
                hidden_dropout=0.1,#10
                extra_tokens_vocab_size=None,#11
                project_position_embeddings=True,#12
                mask_zero=False,#13
                adapter_size=None,#14
                adapter_activation='gelu',#15
                adapter_init_scale=0.001,#16
                num_heads=12,#17
                size_per_head=None,#18
                query_activation=None,#19
                key_activation=None,#20
                value_activation=None,#21
                attention_dropout=0.1,#22
                negative_infinity=-10000.0,#23
                intermediate_size=3072,#24
                intermediate_activation='gelu',#25
                num_layers=12,#26
                out_layer_ndxs=None,#27
                shared_layer=False,#28
                batch_size = 256,
                #获取对应的切分分割字符内容
                *args, **kwargs):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        super(Bert, self).__init__()
        assert maxlen <= max_position_embeddings,"maxlen cannot larger than max_position_embeddings"
        self.maxlen = maxlen#0,使用过，最大长度，后续会指定，默认为128
        #回头看看这里的maxlen能否通过读取相应的数组维度内容进行实现
        self.initializer_range = initializer_range#1
        self.max_position_embeddings = max_position_embeddings#2
        self.embedding_size = embedding_size#4
        self.project_embeddings_with_bias = project_embeddings_with_bias#5
        self.vocab_size = vocab_size#6
        self.use_token_type = use_token_type#7
        self.use_position_embeddings = use_position_embeddings#8
        self.token_type_vocab_size = token_type_vocab_size#9
        self.hidden_dropout = hidden_dropout#10
        self.extra_tokens_vocab_size = extra_tokens_vocab_size#11
        self.project_position_embeddingFs = project_position_embeddings#12
        self.mask_zero = mask_zero#13
        self.adapter_size = adapter_size#14
        self.adapter_activation = adapter_activation#15
        self.adapter_init_scale = adapter_init_scale#16
        self.num_heads = num_heads#17注意力头数，需指定
        assert embedding_size%num_heads == 0,"size_per_head必须能够整除num_heads"
        self.size_per_head = embedding_size//num_heads#18
        self.attention_dropout = attention_dropout#22
        self.negative_infinity = negative_infinity#23
        self.intermediate_size = intermediate_size#24
        self.intermediate_activation = intermediate_activation#25
        self.num_layers = num_layers#26 attention层数，需指定
        self.out_layer_ndxs = out_layer_ndxs#27
        self.shared_layer = shared_layer#28
        self.batch_size = batch_size#最大批次，需指定

    def build(self, input_ids):
        #加一个是否len为2的判断
        if isinstance(input_ids,list):
            assert len(input_ids) == 2
            input_ids_shape,token_type_ids_shape = input_ids
        else:
            input_ids_shape = input_ids
        self.batch_size = input_ids_shape[0]
        self.maxlen = input_ids_shape[1]
        self.embeddings = Embeddings(vocab_size = self.vocab_size,
                                embedding_size = self.embedding_size,
                                mask_zero = self.mask_zero,
                                max_position_embeddings = self.max_position_embeddings,
                                token_type_vocab_size = self.token_type_vocab_size,
                                hidden_dropout = self.hidden_dropout)
        self.encoder_layer = []
        for layer_ndx in range(self.num_layers):
            encoder_layer = Transformer(initializer_range = self.initializer_range,
                                           num_heads = self.num_heads,
                                           size_per_head = self.size_per_head,
                                           attention_dropout = 0.1,
                                           negative_infinity = -10000.0,
                                           intermediate_size = self.intermediate_size
                                          )
            encoder_layer.name = 'transformer_%d'%layer_ndx
            self.encoder_layer.append(encoder_layer)
        super(Bert, self).build(input_ids)
    def call(self,input_ids):
        judge1 = tf.is_tensor(input_ids)
        judge2 = True if type(input_ids) is np.ndarray else False
        judge3 = True if type(input_ids) is list else False
        assert judge1 or judge2 or judge3, "Expecting input to be a tensor or numpy or list type"
        output_embeddings = self.embeddings(input_ids)
        #这里定义为多个transformer的内容
        for encoder_layer in self.encoder_layer:
            output_embeddings = encoder_layer(output_embeddings)
        return output_embeddings
        
class  Embeddings(tf.keras.layers.Layer):
    #实现word_embedding,segment_embedding,position_embedding的相加求和结果
    name = 'embeddings'
    def __init__(self,
                 vocab_size = 30522,
                 embedding_size = 768,
                 mask_zero = False,
                 max_position_embeddings = 512,
                 token_type_vocab_size = 2,
                 initializer_range = 0.02,
                 hidden_dropout = 0.1):
        #之前__init__之中少写了一个self,报错multiple initialize
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.mask_zero = mask_zero
        self.max_position_embeddings = max_position_embeddings
        self.token_type_vocab_size = token_type_vocab_size
        self.initializer_range = initializer_range
        self.hidden_dropout = hidden_dropout

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        
    def build(self, inputs):
        #mask_zero:是否把0看作为一个应该被遮蔽的特殊的padding的值
        #对于可变长的循环神经网络十分有用
        #self.word_embeddings_layer权重矩阵(30522,768)
        if isinstance(inputs,list):
            assert len(inputs) == 2
            input_ids_shape,token_type_ids_shape = inputs
        else:
            input_ids_shape = inputs
        maxlen = input_ids_shape[1]
        #build之中的inputs传入的是TensorShape类型的数据内容
        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim = self.vocab_size,
            output_dim = self.embedding_size,
            mask_zero = self.mask_zero,
            name = "word_embeddings"
        )
        self.position_embeddings_table = self.add_weight(
            name="position_embeddings/embeddings",
            dtype=K.floatx(),
            shape=[self.max_position_embeddings,self.embedding_size],
            initializer=self.create_initializer()
        )

        self.segment_embeddings_layer = keras.layers.Embedding(
            input_dim = self.token_type_vocab_size,
            output_dim = self.embedding_size,
            mask_zero = self.mask_zero,
            name = "segment_embeddings"
        )
        self.layer_normalization = LayerNormalization(name="layer_normalization_first")
        self.dropout_layer = keras.layers.Dropout(rate=self.hidden_dropout)
        #self.segment_embeddings_layer权重矩阵(2,768)
        #看清楚哪里报错，这里报错module 'tensorflow.keras' has no
        #attribute 'layer'
        #把这一部分改编为Embeddings_layer并且传入相应的初始化参数即可
        super(Embeddings,self).build(inputs)
    def call(self,input_ids):
        #assert len(input_ids) > 0,"Bert input length cannot be zero.Please adjust your input"
        #inputs = (None,128)
        #这种情形能够产生(None,128)的对应Tensor
        segment_ids = None
        if isinstance(input_ids,list):
            assert 2 == len(input_ids),"Expecting inputs to be a [input_ids,token_type_ids] list"
            input_ids,segment_ids = input_ids[0],input_ids[1]
        shape = input_ids.shape
        maxlen = shape[1]
        batch_size = shape[0]
        #input_ids = tf.convert_to_tensor(input_ids)
        word_embeddings = self.word_embeddings_layer(input_ids)
        assert_op = tf.compat.v2.debugging.assert_less_equal(maxlen,self.max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            position_embeddings = tf.slice(self.position_embeddings_table,
                                                [0,0],
                                                [maxlen,-1])
        #这里的max_position_embeddings使用切片技术实现的
        #从position_embedding_table之中切出对应的maxlen长度的数据
        currentshape = [1,maxlen,self.embedding_size]
        #与input_ids的形状相同
        if segment_ids == None:
            segment_ids = tf.zeros_like(input_ids)
        #@@@segment_ids = (None,178)
        #@@@segment_ids = (24,178)
        segment_embeddings = self.segment_embeddings_layer(segment_ids)
        #这段后续segment_embeddings要扩充的话，提供几个tensorflow的方法
        #tf.zeros_like,tf.ones_like,tf.pad,tf.concat():tensor连接的方向
        #tf.concat_dim:0表示行，1表示列，比如t1 = [[1,2,3],[4,5,6]]
        #t2 = [[7,8,9],[10,11,12]],tf.concat(0,[t1,t2]) = [[1,2,3],[4,5,6]
        #[7,8,9],[10,11,12]]
        #tf.concat(1,[t1,t2]) = [[1,2,3,7,8,9],[4,5,6,10,11,12]]
        results = word_embeddings+tf.reshape(position_embeddings,currentshape)
        results = results+segment_embeddings
        results = self.layer_normalization(results)
        results = self.dropout_layer(results)
        return results

class  Transformer(tf.keras.layers.Layer):
    name = 'transformer'
    def __init__(self,
                 initializer_range = 0.02,
                 embedding_size = 768,
                 hidden_dropout = 0.1,
                 adapter_size = None,
                 adapter_activation = 'gelu',
                 adapter_init_scale = 0.001,
                 num_heads = 12,
                 size_per_head = 64,
                 query_activation = None,
                 key_activation = None,
                 value_activation = None,
                 attention_dropout = 0.1,
                 negative_infinity = -10000.0,
                 intermediate_size = 3072,
                 **kwargs):
        super(Transformer,self).__init__()
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size
        self.hidden_dropout = hidden_dropout
        self.adapter_size = adapter_size
        self.adapter_activation = adapter_activation
        self.adapter_init_scale = adapter_init_scale
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.query_activation = query_activation
        self.key_activation = key_activation
        self.value_activation = value_activation
        self.attention_dropout = attention_dropout
        self.negative_infinity = negative_infinity
        self.intermediate_size = intermediate_size
    def build(self,input_shape):
        self.attention = AttentionLayer(initializer_range = self.initializer_range,
                                        num_heads = self.num_heads,
                                        size_per_head = self.size_per_head,
                                        query_activation = self.query_activation,
                                        key_activation = self.key_activation,
                                        value_activation = self.value_activation,
                                        attention_dropout = self.attention_dropout,
                                        negative_infinity = self.negative_infinity,
                                        name = "attention")
        self.dense0 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "dense0")
        self.dropout0 = keras.layers.Dropout(rate=self.hidden_dropout)
        self.layer_norm0 = LayerNormalization()
        self.layer_norm0.name = "layer_normalization_0"
        #自己定义的keras.layers层定义名称之后，名称赋值并不能赋值上
        self.dense = keras.layers.Dense(units = self.intermediate_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "dense")
        self.dense1 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "dense1")
        self.dropout1 = keras.layers.Dropout(rate=self.hidden_dropout)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm1.name = "layer_normalization_1"
        super(Transformer,self).build(input_shape)
                                         
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

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

class AttentionLayer(tf.keras.layers.Layer):
    name = 'attention'
    def __init__(self,
                 initializer_range = 0.02,
                 num_heads = 12,
                 size_per_head = 64,
                 query_activation = None,
                 key_activation = None,
                 value_activation = None,
                 attention_dropout = 0.1,
                 negative_infinity = -10000.0,
                 **kwargs):
        super(AttentionLayer,self).__init__()
        self.initializer_range = initializer_range
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.query_activation = query_activation
        self.key_activation = key_activation
        self.value_activation = value_activation
        self.attention_dropout = attention_dropout
        self.negative_infinity = negative_infinity
        
        self.query_layer = None
        self.key_layer = None
        self.value_layer = None
        
        self.supports_masking = True
        self.initializer_range = 0.02
    
    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
    
    def build(self,input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        #build之中可以通过input_spec来定义对应新的keras.layers.InputSpec
        dense_units = self.num_heads*self.size_per_head
        #768=12*64,12头，每一头64个维度
        self.query_layer = keras.layers.Dense(units=dense_units,activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer = keras.layers.Dense(units=dense_units,activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="key")
        self.value_layer = keras.layers.Dense(units=dense_units,activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.attention_dropout)
        super(AttentionLayer,self).build(input_shape)

    def compute_output_shape(self,input_shape):
        from_shape = input_shape
        output_shape = [from_shape[0],from_shape[1],self.num_heads*self.size_per_head]
        return output_shape
        
    def call(self,inputs,**kwargs):
        input_shape = tf.shape(input=inputs)
        batch_size,seq_len,width = input_shape[0],input_shape[1],input_shape[2]
        def transpose_for_scores(input_tensor,seq_len):
        #改变数组形状，由[batch_size,from_seq_len,num_heads*size_per_head]
        #->[batch_size,num_heads,from_seq_len,size_per_head]
            output_shape = [batch_size,seq_len,
                           self.num_heads,self.size_per_head]
            #input_tensor = ("bert/transformer/attention_layer/query/
            #BiasAdd:0",shape=(None,128,768))
            #output_shape = [None,128,12,64]
            #output_shape = [tf.Tensor,shape=(),tf.Tensor,shape=(),12,64]
            output_tensor = K.reshape(input_tensor,output_shape)
            #小函数能够使用外面的变量
            #注意这里的output_shape必须放入tensor类型的数值
            return tf.transpose(a=output_tensor,perm=[0,2,1,3])
        query = self.query_layer(inputs)
        #query = (None,128,768)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        #query = (None,128,768)
        query = transpose_for_scores(query,seq_len)
        key = transpose_for_scores(key,seq_len)
        #(None,12,128,64)
        #query = (None,12,None,64),key = (None,12,None,64)
        attention_scores = tf.matmul(query,key,transpose_b=True)
        #(None,12,128,64)*(None,12,64,128) = (None,12,64,64)
        attention_scores = attention_scores/tf.sqrt(float(self.size_per_head))
        #attention_scores = (None,12,None,None)
        #attention_scores = (32,12,178,178)
        attention_probs = tf.nn.softmax(attention_scores)
        #value = (None,12,None,64)
        value = tf.reshape(value,[batch_size,seq_len,
                                  self.num_heads,self.size_per_head])
        value = tf.transpose(a=value,perm=[0,2,1,3])
        #value = (None,12,None,64)
        context_layer = tf.matmul(attention_probs,value)
        #(None,12,None,None)*(None,12,None,64) = (None,12,None,64)
        #(None,12,64,64)*(None,12,64,128) = (None,12,64,128)
        #context_layer = (None,12,None,768)
        context_layer = tf.transpose(a=context_layer,perm=[0,2,1,3])
        #(None,64,12,128)
        #context_layer = (None,12,None,64)
        output_shape = [batch_size,seq_len,
                       self.num_heads*self.size_per_head]
        #output_shape = [None,None,12,64]
        context_layer = tf.reshape(context_layer,output_shape)
        #context_layer = (None,128,12,64)
        #final_context_layer = (None,None,12,64)
        return context_layer

class LayerNormalization(tf.keras.layers.Layer):
    name = 'layer_normalization'
    def __init__(self,
                 **kwargs):
        super(LayerNormalization,self).__init__()
        self.gamma = None
        self.beta  = None
        self.supports_masking = True
        self.epsilon = 1e-12

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