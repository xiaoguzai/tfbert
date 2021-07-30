import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import os
import numpy as np
from tensorflow_addons.text.crf import crf_log_likelihood
#import tensorflow.keras.layers.Layer as Layer
#No module named 'tensorflow.keras.layers.Layer'
#之前的写法是import tensorflow.keras.layers.Layer as Layer报错
class Bert(tf.keras.layers.Layer):
    name = 'nezha'
    def __init__(self,
                maxlen = 128,#0
                initializer_range=0.02,#1
                max_position_embeddings=512,#2
                embedding_size=768,#4
                project_embeddings_with_bias=True,#5
                vocab_size=21128,#6
                hidden_dropout=0.1,#10
                extra_tokens_vocab_size=None,#11
                project_position_embeddings=True,#12
                mask_zero=False,#13
                adapter_size=None,#14
                hidden_act='gelu',#15
                adapter_init_scale=0.001,#16
                num_attention_heads=12,#17
                size_per_head=None,#18
                attention_probs_dropout_prob=0.1,#22
                negative_infinity=-10000.0,#23
                intermediate_size=3072,#24
                intermediate_activation='gelu',#25
                num_layers=12,#26
                batch_size = 256,
                #获取对应的切分分割字符内容
                directionality = 'bidi',
                pooler_fc_size = 768,
                pooler_num_attention_heads = 12,
                pooler_num_fc_layers = 3,
                pooler_size_per_head = 128,
                pooler_type = "first_token_transform",
                type_vocab_size = 2,
                with_mlm = False,
                mlm_activation = 'softmax',
                mode = 'bert',
                solution = 'seq2seq',
                max_relative_position = 64,
                with_pooler = True,
                *args, **kwargs):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        #super(Nezha, self).__init__()
        super(Bert,self).__init__()
        print('__init__ Nezha')
        assert maxlen <= max_position_embeddings,"maxlen cannot larger than max_position_embeddings"
        self.maxlen = maxlen#0,使用过，最大长度，后续会指定，默认为128
        #回头看看这里的maxlen能否通过读取相应的数组维度内容进行实现
        self.initializer_range = initializer_range#1
        self.max_position_embeddings = max_position_embeddings#2
        self.embedding_size = embedding_size#4
        self.project_embeddings_with_bias = project_embeddings_with_bias#5
        self.vocab_size = vocab_size#6
        self.token_type_vocab_size = 2#9
        self.hidden_dropout = hidden_dropout#10
        self.extra_tokens_vocab_size = extra_tokens_vocab_size#11
        self.project_position_embeddings = project_position_embeddings#12
        self.mask_zero = mask_zero#13
        self.adapter_size = adapter_size#14
        self.hidden_act = hidden_act#15
        self.adapter_init_scale = adapter_init_scale#16
        self.num_attention_heads = num_attention_heads#17注意力头数，需指定
        assert embedding_size%num_attention_heads == 0,"size_per_head必须能够整除num_attention_heads"
        self.size_per_head = embedding_size//num_attention_heads#18
        self.attention_probs_dropout_prob = attention_probs_dropout_prob#22
        self.negative_infinity = negative_infinity#23
        self.intermediate_size = intermediate_size#24
        self.intermediate_activation = intermediate_activation#25
        self.num_layers = num_layers#26 attention层数，需指定
        self.batch_size = batch_size#最大批次，需指定
        self.directionality = directionality
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.with_mlm = with_mlm
        self.mlm_activation = mlm_activation
        self.mode = mode
        self.solution = solution
        self.max_relative_position = max_relative_position
        self.with_pooler = with_pooler

    def get_config(self):
        config = super(Nezha,self).get_config()
        dicts = {
            "name":"nezha",
            "maxlen":self.maxlen,#0,使用过，最大长度，后续会指定，默认为128
        #回头看看这里的maxlen能否通过读取相应的数组维度内容进行实现
            "initializer":self.initializer_range,#1
            "max_position_embeddings":self.max_position_embeddings,#2
            "embedding_size":self.embedding_size,#4
            "project_embeddings_with_bias":self.project_embeddings_with_bias,#5
            "vocab_size":self.vocab_size,#6
            "token_type_vocab_size":self.token_type_vocab_size,#9
            "hidden_dropout":self.hidden_dropout,#10
            "extra_tokens_vocab_size":self.extra_tokens_vocab_size,#11
            "project_position_embeddings":self.project_position_embeddings,#12
            "mask_zero":self.mask_zero,#13
            "adapter_size":self.adapter_size,#14
            "hidden_act":self.hidden_act,#15
            "adapter_init_scale":self.adapter_init_scale,#16
            "num_attention_heads":self.num_attention_heads,#17注意力头数，需指定
            "size_per_head":self.size_per_head,#18
            "attention_probas_dropout_prob":self.attention_probs_dropout_prob,#22
            "negative_infinity":self.negative_infinity,#23
            "intermediate_size":self.intermediate_size,#24
            "intermediate_activation":self.intermediate_activation,#25
            "num_layers":self.num_layers,#26 attention层数，需指定
            "batch_size":self.batch_size,#最大批次，需指定
            "directionality":self.directionality,
            "pooler_fc_size":self.pooler_fc_size,
            "pooler_num_attention_heads":self.pooler_num_attention_heads,
            "pooler_num_fc_layers":self.pooler_num_fc_layers,
            "pooler_size_per_head":self.pooler_size_per_head,
            "pooler_type":self.pooler_type,
            "with_mlm":self.with_mlm,
            "mlm_activation":self.mlm_activation,
            "mode":self.mode,
            "solution":self.solution,
            "max_relative_position":self.max_relative_position
        }
        config.update(dicts)
        return config

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
                                           num_attention_heads = self.num_attention_heads,
                                           embedding_size = self.embedding_size,
                                           size_per_head = self.size_per_head,
                                           attention_probs_dropout_prob = 0.1,
                                           negative_infinity = -10000.0,
                                           intermediate_size = self.intermediate_size,
                                           mode = self.mode,
                                           solution = self.solution,
                                           max_relative_position = self.max_relative_position
                                          )
            encoder_layer.name = 'transformer_%d'%layer_ndx
            self.encoder_layer.append(encoder_layer)
        if self.with_pooler:
            self.pooler = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "pooler")
           
        if self.with_mlm:
            self.mlm_dense0 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        activation = self.get_activation('gelu'),
                                        name = "mlm_dense0")
            self.mlm_norm = LayerNormalization()
            self.mlm_norm.name = 'mlm_norm'
            self.mlm_dense1 = keras.layers.Dense(units = self.vocab_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "mlm_dense1")
            #这里使用add_weight作为bias进行操作
            #self.mlm_activation_layer = keras.layers.Activation(self.mlm_activation)
        super(Bert, self).build(input_ids)
    
    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
    
    def get_activation(self,activation_string):
        if not isinstance(activation_string, str):
            return activation_string

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return self.gelu
        elif act == "gelu_exact":
            return self.gelu_exact
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)
    
    def gelu(self,x):
        """
        Gelu activation from arXiv:1606.08415.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
        ))
        return x * cdf

    def gelu_exact(self,x):
        return x * tf.math.erfc(-x / tf.sqrt(2.)) / 2.
    
    def call(self,input_ids):
        judge1 = tf.is_tensor(input_ids)
        judge2 = True if type(input_ids) is np.ndarray else False
        judge3 = True if isinstance(input_ids,list) else False
        assert judge1 or judge2 or judge3,"Expecting input to be a tensor or numpy or list type"
        output_embeddings = self.embeddings(input_ids)
        #output_embeddings = [(None,128,768),(None,128)]
        #output_embeddings = [result,segment_embeddings]
        
        for encoder_layer in self.encoder_layer:
            output_embeddings = encoder_layer(output_embeddings)
        
        outputs = output_embeddings[0]
        outputs = self.pooler(outputs)
        #transformer outputs = (None,128,768)
        if self.with_mlm:
            outputs = self.mlm_dense0(outputs)
            outputs = self.mlm_norm(outputs)
            #print('@@@@@@output0 = @@@@@@')
            #print(outputs)
            outputs = self.mlm_dense1(outputs)
            #print('------outputs1 = ------')
            #print(outputs)
            if self.mlm_activation == 'softmax':
                outputs = keras.activations.softmax(outputs,axis=-1)
                #print('``````outputs2 = ``````')
                #print(outputs)
        return outputs
        
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

    def get_config(self):
        config = super(Embeddings,self).get_config()
        dicts = {
            "vocab_size":self.vocab_size,
            "embedding_size":self.embedding_size,
            "mask_zero":self.mask_zero,
            "max_position_embeddings":self.max_position_embeddings,
            "token_type_vocab_size":self.token_type_vocab_size,
            "initializer_range":self.initializer_range,
            "hidden_dropout":self.hidden_dropout
        }
        config.update(dicts)
        return config

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

    def build(self, inputs):
        #mask_zero:是否把0看作为一个应该被遮蔽的特殊的padding的值
        #对于可变长的循环神经网络十分有用
        #self.word_embeddings_layer权重矩阵(30522,768)
        
        #初始化之后，第二次还会调用build函数吗？？？
        #哪里导致数组的形状不一致？这里导致每一次的参数都需要传入一次？？？
        if isinstance(inputs,list):
            assert len(inputs) == 2
            input_ids_shape,token_type_ids_shape = inputs
        else:
            input_ids_shape = inputs
        #input_ids = (None,128),token_type_ids = (None,128)
        maxlen = input_ids_shape[1]
        #build之中的inputs传入的是TensorShape类型的数据内容
        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim = self.vocab_size,
            output_dim = self.embedding_size,
            mask_zero = self.mask_zero,
            name = "word_embeddings"
        )
        
        self.segment_embeddings_layer = keras.layers.Embedding(
            input_dim = self.token_type_vocab_size,
            output_dim = self.embedding_size,
            mask_zero = self.mask_zero,
            name = "segment_embeddings"
        )
        self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="layer_normalization")
        #self.layer_normalization = LayerNormalization(name="layer_normalization_first")
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
        #当为list数组形式的时候可以手动传入对应的segment_ids
        shape = input_ids.shape
        #print('shape = ')
        #print(shape)
        maxlen = shape[1]
        batch_size = shape[0]
        #input_ids = tf.convert_to_tensor(input_ids)
        word_embeddings = self.word_embeddings_layer(input_ids)

        currentshape = [1,maxlen,self.embedding_size]
        #与input_ids的形状相同
        if segment_ids == None:
            segment_ids = tf.zeros_like(input_ids)
        #@@@segment_ids = (None,178)
        #@@@segment_ids = (24,178)
        segment_embeddings = self.segment_embeddings_layer(segment_ids)
        
        results = word_embeddings+segment_embeddings
        results = self.layer_normalization(results)
        results = self.dropout_layer(results)
        #results = (None,128,768),segment_embeddings = (None,128,768)
        return [results,segment_ids]
        #!!!这里只需要segment_ids作为后面的掩码使用，而不需要对应的segment_embeddings

class  Transformer(tf.keras.layers.Layer):
    name = 'transformer'
    def __init__(self,
                 initializer_range = 0.02,
                 embedding_size = 768,
                 hidden_dropout = 0.1,
                 adapter_size = None,
                 hidden_act = 'gelu',
                 adapter_init_scale = 0.001,
                 num_attention_heads = 12,
                 size_per_head = 64,
                 query_activation = None,
                 key_activation = None,
                 value_activation = None,
                 attention_probs_dropout_prob = 0.1,
                 negative_infinity = -10000.0,
                 intermediate_size = 3072,
                 mode = 'bert',
                 solution = 'seq2seq',
                 max_relative_position = 64,
                 **kwargs):
        super(Transformer,self).__init__()
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size
        self.hidden_dropout = hidden_dropout
        self.adapter_size = adapter_size
        self.hidden_act = hidden_act
        self.adapter_init_scale = adapter_init_scale
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_activation = query_activation
        self.key_activation = key_activation
        self.value_activation = value_activation
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.negative_infinity = negative_infinity
        self.intermediate_size = intermediate_size
        self.mode = mode
        self.solution = solution
        self.max_relative_position = max_relative_position
        
    def get_config(self):
        config = super(Transformer,self).get_config()
        dicts = {
            "initializer_range":self.initializer_range,
            "embedding_size":self.embedding_size,
            "hidden_dropout":self.hidden_dropout,
            "adapter_size":self.adapter_size,
            "hidden_act":self.hidden_act,
            "adapter_init_scale":self.adapter_init_scale,
            "num_attention_heads":self.num_attention_heads,
            "size_per_head":self.size_per_head,
            "query_activation":self.query_activation,
            "key_activation":self.key_activation,
            "value_activation":self.value_activation,
            "attention_probs_dropout_prob":self.attention_probs_dropout_prob,
            "negative_infinity":self.negative_infinity,
            "intermediate_size":self.intermediate_size,
            "mode":self.mode,
            "solution":self.solution,
            "max_relative_position":self.max_relative_position
        }
        config.update(dicts)
        return config

    def build(self,input_shape):
        self.attention = AttentionLayer(initializer_range = self.initializer_range,
                                        num_attention_heads = self.num_attention_heads,
                                        size_per_head = self.size_per_head,
                                        query_activation = self.query_activation,
                                        key_activation = self.key_activation,
                                        value_activation = self.value_activation,
                                        attention_probs_dropout_prob = self.attention_probs_dropout_prob,
                                        negative_infinity = self.negative_infinity,
                                        #name = "attention",
                                        mode = self.mode,
                                        solution = self.solution,
                                        max_relative_position = self.max_relative_position)
        self.dense0 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        #kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                        name = "dense0")
        self.dropout0 = keras.layers.Dropout(rate=self.hidden_dropout)
        #self.layer_norm0 = LayerNormalization()
        #self.layer_norm0.name = "layer_normalization_0"
        #自己定义的keras.layers层定义名称之后，名称赋值并不能赋值上
        self.layer_norm0 = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="layer_normalization_0")
        self.dense = keras.layers.Dense(units = self.intermediate_size,
                                        kernel_initializer = self.create_initializer(),
                                        #kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                        activation = self.get_activation('gelu'),
                                       #activation = 'gelu',
                                        name = "dense")
        #错误点1：这里的activation少加了一个激活的gelu函数
        self.dense1 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        #kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                        name = "dense1")
        self.dropout1 = keras.layers.Dropout(rate=self.hidden_dropout)
        #self.layer_norm1 = LayerNormalization()
        #self.layer_norm1.name = "layer_normalization_1"
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="layer_normalization_1")
        super(Transformer,self).build(input_shape)
                                        
    def call(self,inputs,**kwargs):
        residual = inputs[0]
        embedding_output = self.attention(inputs)
        #transformer结构图中确实是residual inputs在attention之前
        #然而这里进行分类任务的时候实测residual inputs在attention之后效果更好
        #residual = inputs
        embedding_output = self.dense0(embedding_output)
        embedding_output = self.dropout0(embedding_output)
        #print('after dropout')
        #print('embedding_output = ')
        #print(embedding_output)
        embedding_output = self.layer_norm0(residual+embedding_output)
        #print('after layer_norm0')
        #print('embedding_output = ')
        #print(embedding_output)
        residual = embedding_output
        embedding_output = self.dense(embedding_output)
        #self.dense对应着feed forward层
        #print('after intermediate dense')
        #print('embedding_output = ')
        #print(embedding_output)
        #embedding_output = self.gelu(embedding_output)
        embedding_output = self.dense1(embedding_output)
        embedding_output = self.dropout1(embedding_output)
        embedding_output = self.layer_norm1(residual+embedding_output)
        return [embedding_output,inputs[1]]

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

    def gelu(self,x):
        cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))
        return x * cdf

    def gelu_exact(self,x):
        return x * tf.math.erfc(-x / tf.sqrt(2.)) / 2.

    def get_activation(self,activation_string):
        if not isinstance(activation_string, str):
            return activation_string

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return self.gelu
        elif act == "gelu_exact":
            return self.gelu_exact
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)

class AttentionLayer(tf.keras.layers.Layer):
    name = 'attention'
    def __init__(self,
                 initializer_range = 0.02,
                 num_attention_heads = 12,
                 size_per_head = 64,
                 query_activation = None,
                 key_activation = None,
                 value_activation = None,
                 attention_probs_dropout_prob = 0.1,
                 negative_infinity = -10000.0,
                 mode = 'bert',
                 solution = 'seq2seq',
                 max_relative_position = 64,
                 **kwargs):
        super(AttentionLayer,self).__init__()
        self.initializer_range = initializer_range
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_activation = query_activation
        self.key_activation = key_activation
        self.value_activation = value_activation
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.negative_infinity = negative_infinity
        
        self.query_layer = None
        self.key_layer = None
        self.value_layer = None
        
        self.supports_masking = True
        self.initializer_range = 0.02
        self.mode = mode
        self.solution = solution
        self.maxlen = None
        self.max_relative_position = max_relative_position
        
        
    def get_config(self):
        config = super(AttentionLayer,self).get_config()
        dicts = {
            "initializer_range":self.initializer_range,
            "num_attention_heads":self.num_attention_heads,
            "size_per_head":self.size_per_head,
            "query_activation":self.query_activation,
            "key_activation":self.key_activation,
            "value_activation":self.value_activation,
            "attention_probs_dropout_prob":self.attention_probas_dropout_prob,
            "negative_infinity":self.negative_infinity,
            "query_layer":self.query_layer,
            "key_layer":self.key_layer,
            "value_layer":self.value_layer,
            
            "supports_masking":self.supports_masking,
            "initializer_range":self.initializer_range,
            "mode":self.mode,
            "solution":self.solution,
            "max_relative_position":self.max_relative_position
        }
        config.update(dicts)
        return config
    
    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
    
    def build(self,input_ids):
        if isinstance(input_ids,list):
            assert len(input_ids) == 2,"Expecting inputs to be a [input_ids,token_type_ids] list"
            input_ids_shape,token_type_ids_shape = input_ids
        else:
            input_ids_shape = input_ids
        self.maxlen = input_ids_shape[1]
        dense_units = self.num_attention_heads*self.size_per_head
        #768=12*64,12头，每一头64个维度
        self.query_layer = keras.layers.Dense(units=dense_units,activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              #kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                              name="query")
        self.key_layer = keras.layers.Dense(units=dense_units,activation=self.key_activation,
                                              kernel_initializer=self.create_initializer(),
                                              #kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                              name="key")
        self.value_layer = keras.layers.Dense(units=dense_units,activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              #kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.attention_probs_dropout_prob)
        self.maxlen = input_ids_shape[1]
        super(AttentionLayer,self).build(input_ids)

    def compute_output_shape(self,input_shape):
        from_shape = input_shape
        output_shape = [from_shape[0],from_shape[1],self.num_attention_heads*self.size_per_head]
        return output_shape
        
    def seq2seq_compute_attention_bias(self,s):
        idxs = K.cumsum(s, axis=1)
        mask =idxs[:, None, :] <= idxs[:, :, None]
        mask = K.cast(mask, K.floatx())
        return -(1-mask) * 1e12
    
    def lefttoright_compute_attention_bias(self,s):
        seq_len = K.shape(s)[1]
        idxs = K.arange(0, seq_len)
        mask = idxs[None, :] <= idxs[:, None]
        mask = K.cast(mask, K.floatx())
        return -(1-mask) * 1e12

    def _generate_relative_positions_matrix(self,length, max_relative_position, cache=False):
        if not cache:
            #range_vec = tf.range(length)
            range_vec = K.arange(0,length,dtype='int32')
            range_mat = K.reshape(tf.tile(range_vec, [length]), [length, length])
            distance_mat_clipped = range_mat - tf.transpose(range_mat)
            distance_mat_clipped = tf.clip_by_value(distance_mat_clipped,-max_relative_position,
                                                max_relative_position)
            r"""
            distance_mat_clipped = 
            tensor([[0,1,2,...64,64,64],
                    [-1,0,1,...64,64,64],
                    .....................
                    [-64,-64,..-2,-1,0]]
            """
        else:
            distance_mat = tf.expand_dims(tf.range(-length+1, 1, 1), 0)
            distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        r"""
        final_mat = 
        tensor([[64,65,...128,128],
                [63,64,...128,128],
                ..................
                [0,0,0,...63,64,65],
                [0,0,0,...62,63,64]])
        """
        return final_mat


    def relative_positions_calculate(self,length, depth, max_relative_position, cache=False):
        relative_positions_matrix = self._generate_relative_positions_matrix(
            length, max_relative_position, cache=cache)
        vocab_size = max_relative_position * 2 + 1
        embeddings_table = np.zeros([vocab_size, depth]) #range(vocab_size * depth)#tf.get_variable(name="embeddings", shape=[vocab_size, depth], initializer=create_initializer())
        r"""
        embeddings_table = 
        [[0,0,...0,0,0],
         [0,0,...0,0,0],
         ..............
         [0,0,...0,0,0],
         [0,0,...0,0,0]],shape = [129,64]
        """
        position = K.arange(0.0, vocab_size, 1.0)#.unsqueeze(1)
        position = K.reshape(position, [vocab_size, -1])
        for pos in range(vocab_size):
            for i in range(depth // 2):
                embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
                embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))
        embeddings_table_tensor = tf.convert_to_tensor(embeddings_table, tf.float32)
        flat_relative_positions_matrix = tf.reshape(relative_positions_matrix, [-1])
        #将上面embeddings_table = (512,512)展平为([262144])
        one_hot_relative_positions_matrix = tf.one_hot(flat_relative_positions_matrix, depth=vocab_size)
        #变成[262144,vocab_size]
        embeddings = tf.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
        #my_shape = relative_positions_matrix.shape.as_list()
        #[262144,vocab_size]*[vocab_size,depth] = [262144,depth]
        my_shape = tf.shape(relative_positions_matrix)
        my_shape1,my_shape2 = my_shape[0],my_shape[1]
        #my_shape = Tensor("nezha/.../Shape_1:",shape=(2,),dtype=int32)
        my_shape = [my_shape1,my_shape2,depth]
        embeddings = tf.reshape(embeddings, my_shape)
        #[262144,depth(64)]变成[512,512,64]
        return embeddings
    
    def call(self,inputs,**kwargs):
       #!!!这里报错:Layer attention expects 1 input(s),but it received 2 input tensors
       #!!!遥想到上面的Embedding放了两个Tensor却没有被报错,所以这段可以仿照上面的Embedding对应
       #!!!层进行书写相应内容
        if isinstance(inputs,list):
            assert len(inputs) == 2,"Expecting inputs to be a [input_ids,token_type_ids] list"
            input_ids,segment_ids = inputs[0],inputs[1]
        else:
            input_ids_shape = inputs[0]
        #input_shape = tf.shape(input=inputs[0])
        #这里get_shape()可以将tensor形状以list展示
        input_shape = tf.shape(input=inputs[0])
        batch_size,seq_len,width = input_shape[0],input_shape[1],input_shape[2]
        self.maxlen = seq_len
        #!!!这里必须使用tf.shape，不然会报错
        def transpose_for_scores(input_tensor,seq_len):
        #改变数组形状，由[batch_size,from_seq_len,num_attention_heads*size_per_head]
        #->[batch_size,num_attention_heads,from_seq_len,size_per_head]
            output_shape = [batch_size,seq_len,
                           self.num_attention_heads,self.size_per_head]
            #input_tensor = ("bert/transformer/attention_layer/query/
            #BiasAdd:0",shape=(None,128,768))
            #output_shape = [None,128,12,64]
            #output_shape = [tf.Tensor,shape=(),tf.Tensor,shape=(),12,64]
            output_tensor = K.reshape(input_tensor,output_shape)
            #小函数能够使用外面的变量
            #注意这里的output_shape必须放入tensor类型的数值
            return tf.transpose(a=output_tensor,perm=[0,2,1,3])
        #attention inputs = (1,7,768)
        query = self.query_layer(input_ids)
        #query = (None,128,768)
        key = self.key_layer(input_ids)
        value = self.value_layer(input_ids)
        #query = (None,128,768)
        #query = (None,None,768),key = (None,None,768),value = (None,None,768)
        query = transpose_for_scores(query,seq_len)
        key = transpose_for_scores(key,seq_len)
        #(None,12,128,64),batch_size = 128
        #query = (None,12,None,64),key = (None,12,None,64)
        attention_scores = tf.matmul(query,key,transpose_b=True)
        #(None,12,128,64)*(None,12,64,128) = (None,12,128,128)
        #实现将批次在矩阵之中提取出来的操作
        self.relative_positions_encoding = self.relative_positions_calculate(length=self.maxlen,
                                                                            depth=self.size_per_head,
                                                                            max_relative_position=self.max_relative_position)
        self.relative_positions_encoding = self.relative_positions_encoding[:self.maxlen,:self.maxlen,:]
        #接下来计算一波Q*\alpha{ij},self.relative_positions_encoding=[maxlen,maxlen,64]
        #query = tf.transpose(a=query,perm=[2,1,0,3])
        query = tf.transpose(a=query,perm=[2,0,1,3])

        #query = (None,12,None(maxlen),64)<=>(None(maxlen),12,None,64)
        #query = (512,12,32,64)
        query = K.reshape(query,[self.maxlen,batch_size*self.num_attention_heads,self.size_per_head])
        #query = (None(maxlen),12*None,64),position = (maxlen,maxlen,64)
        #query = (512,384,64)
        current_results = tf.matmul(query,tf.transpose(a=self.relative_positions_encoding,perm=[0,2,1]))
        #attention_scores = (None(maxlen),12*None,64)*(maxlen,64,maxlen) = (maxlen,12*None,maxlen)
        current_results = tf.transpose(a=current_results,perm=[1,0,2])
        #current_results = (12*None,maxlen,maxlen)
        current_results = K.reshape(current_results,[batch_size,self.num_attention_heads,seq_len,seq_len])
        attention_scores = attention_scores+current_results
        attention_scores = attention_scores/tf.sqrt(float(self.size_per_head))
        #attention_scores = (None,12,128,128)
        if self.mode == 'unilm':
            if self.solution == 'seq2seq':
                bias_data = self.seq2seq_compute_attention_bias(segment_ids)
                #当批次为128的时候，bias_data = (1,128,128)
                #attention_scores = attention_scores+bias_data[:,None,:,:]
                attention_scores = attention_scores+bias_data[:,None,:,:]
                #(5,12,128,128)+(5,128,128) = (5,12,128,128)
            elif self.solution == 'lefttoright':
                bias_data = self.lefttoright_compute_attention_bias(segment_ids)
                attention_scores = attention_scores+bias_data
        attention_probs = tf.nn.softmax(attention_scores)
        value = tf.reshape(value,[batch_size,seq_len,
                                  self.num_attention_heads,self.size_per_head])
        value = tf.transpose(a=value,perm=[0,2,1,3])
        #value = (5,12,128,64)
        context_layer = tf.matmul(attention_probs,value)
        attention_scores = attention_probs
        #!!!!!!!!!!!!!!!!!!!!!!!上面这句之前运算的时候遗漏!!!!!!!!!!!!!!!!!!!!!
        #(5,12,128,128)*(5,12,128,64) = (5,12,128,64)
        #context_layer = (batch_size,self.num_attention_heads,seq_len,self.num_attention_heads]
        #目前self.relative_positions_encoding = [maxlen,maxlen,64]
        attention_scores = tf.transpose(a=attention_scores,perm=[2,0,1,3])
        #attention_scores = (128,None,12,128)
        attention_scores = K.reshape(attention_scores,[seq_len,batch_size*self.num_attention_heads,seq_len])
        attention_scores = tf.matmul(attention_scores,self.relative_positions_encoding)
        #(128,None*12,128)*(128,128,64) = (128,None*12,64)
        attention_scores = tf.transpose(a=attention_scores,perm=[1,0,2])
        #(None*12,128,64)
        attention_scores = K.reshape(attention_scores,[batch_size,self.num_attention_heads,seq_len,self.size_per_head])
        context_layer = attention_scores+context_layer
        context_layer = tf.transpose(a=context_layer,perm=[0,2,1,3])
        #(None,64,12,128)
        #context_layer = (None,12,None,64)
        output_shape = [batch_size,seq_len,
                       self.num_attention_heads*self.size_per_head]
        #output_shape = [None,None,12,64]
        context_layer = tf.reshape(context_layer,output_shape)
        #最终转为context_layer = (5,128,12,64)
        #return inputs[0]
        return context_layer
        #这里返回值写错了，之前为return inputs[0]的时候一直报错

class LayerNormalization(tf.keras.layers.Layer):
    name = 'layer_normalization'
    def __init__(self,
                 **kwargs):
        super(LayerNormalization,self).__init__()
        self.gamma = None
        self.beta  = None
        self.supports_masking = True
        self.epsilon = 1e-12
    
    def get_config(self):
        config = super(LayerNormalization,self).get_config()
        dicts = {
            "gamma":self.gamma,
            "beta":self.beta,
            "supports_masking":self.supports_masking,
            "epsilon":self.epsilon
        }
        config.update(dicts)
        return config
    
    def build(self, input_shape):
        #上面这句去除掉会引发对应的输入错误???
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
        outputs = inputs
        outputs = outputs-mean
        std = K.sqrt(var+self.epsilon)
        outputs = outputs/std*self.gamma+self.beta
        #self.gamma和self.beta没有赋值上去
        inv = self.gamma * tf.math.rsqrt(var + self.epsilon)
        res = x * tf.cast(inv, x.dtype) + tf.cast(self.beta - mean * inv, x.dtype)
        #tf.cast()函数执行tensorflow中张量数据类型转换，转换为x.dtype类型
        #这里面使用的res = (inputs-平均)*(1/根号(方差+1e-12))
        #使用的残差的原始函数为(x-平均)*(1/根号(方差+1e-12))
        return res

class ConditionalRandomField(keras.layers.Layer):
    #纯Keras实现CRF层
    #CRF层本质上是一个带训练参数的loss计算层。
    
    def __init__(self, lr_multiplier=1, **kwargs):
        super(ConditionalRandomField, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数

    def build(self, input_shape):
        super(ConditionalRandomField, self).build(input_shape)
        output_dim = input_shape[-1]
        #output_dim = 7
        self._trans = self.add_weight(
            name='trans',
            shape=(output_dim, output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        #使用glorot_uniform进行权重初始化方法:It draws samples from
        #a uniform distribution within [-limit,limit]
        #limit = sqrt(6/(fan_{in}+fan_{out}))
        #fan_{in} = channels_{in}*kernel_{width}*kernel_{height}
        #fan_{out} = channels_{in}*kernel_{width}*kernel_{height}
        #输入网络为28*28*1的数据，卷积核为3*3，卷积核通道数有32个(即输出的通道数
        #有32个，则此时limit = sqrt(6/(3*3*1+3*3*32))
        if self.lr_multiplier != 1:
            K.set_value(self._trans, K.eval(self._trans) / self.lr_multiplier)
        #self._trans:要设置为新值的Tensor,value:将张量设置为Numpy数组(具有相同形状的值)
        #!!!相当于self._trans = K.eval(self._trans)/self.lr_multiplier!!!
    @property
    def trans(self):
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._trans
        else:
            return self._trans

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        #目前inputs = (None,None,7),mask = None
        #inputs = Tensor(shape=(?,?,7),dtype=float32)
        #mask = Tensor("Transformer-11-Forward-Add/All:0",shape=(?,?)
        #dtype = bool)
        #return sequence_masking(inputs, mask, '-inf', 1)
        return inputs

    def target_score(self, y_true, y_pred):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        """
        point_score = tf.einsum('bni,bni->b', y_true, y_pred)  # 逐标签得分
        #point_score = tf.Tensor([-27.828606 -27.828606],shape=(2,))
        point1 = tf.multiply(y_true,y_pred)
        sum = tf.reduce_sum(point1,axis=-1)
        sum = tf.reduce_sum(sum,axis=-1)
        trans_score = tf.einsum(
            'bni,ij,bnj->b', y_true[:, :-1], self.trans, y_true[:, 1:]
        )  # 标签转移得分
        return point_score + trans_score
        #loss返回的应该为一个具体的数值

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        inputs为之前输入的内容，states为上面的网络层输出的结果
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        #inputs = (None,45)
        #mask = (None,1)
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        #states = (None,45,1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        #trans = (1,45,45)
        outputs = tf.reduce_logsumexp(
            states + trans, 1
        )  # (batch_size, output_dim)
        #states+trans = (None,45,1)+(1,45,45) = (None,45,45)
        #经历过tf.reduce_logsumexp之后转变为(None,45)
        #outputs = (None,45)
        outputs = outputs + inputs
        #outputs = (None,45)
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        #outputs = (None,45)
        return outputs, [outputs]

    def dense_loss(self, y_true, y_pred):
        mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
        mask = K.cast(mask, K.floatx())
        y_true, y_pred = y_true * mask, y_pred * mask
        r"""
        如果概率值低于-1e6的部分，对应的y_true的标记和相应的概率值都会被遮盖掉
        """
        target_score = self.target_score(y_true, y_pred)
        # 递归计算log Z
        # target_score = tf.Tensor([5.3490796,5.3490796],shape=(2,))
        init_states = [y_pred[:, 0]]
        y_pred = K.concatenate([y_pred, mask], axis=2)
        #y_pred = (2,50,8),每一个位置数值后面加上一个1,
        #[ 0.63680947 -0.7012377 -0.14899871 -0.1371142 -0.22552827 -0.40690875 0.297494 1]
        input_length = K.int_shape(y_pred[:, 1:])[1]
        #input_length = 49,本身y_pred = (2,50,8),取后面的部分y_pred[:,1:]之后y_pred = (2,50,7)
        #K.int_shape(y_pred[:,1:])[1] = [2,49,8],取[1]之后值为8
        #init_states.shape = (2,7),array([[0.29086065,-0..3617367,...]
        #[0.29086065,-0.03617367,...]],dtype=float32
        #self.log_norm_step = <bound method ConditionalRandomField.log_norm_step
        #of <models.ConditionalRandomField object at 0x7fb460af60d0>>
        #y_pred[:,1:] = (2,49,8)
        #init_states = (2,7)
        #input_length = 49
        log_norm, _, _ = K.rnn(
            self.log_norm_step,
            y_pred[:, 1:],
            init_states,
            input_length=input_length
        )  # 最后一步的log Z向量
        log_norm = tf.reduce_logsumexp(log_norm, 1)  # logsumexp得标量
        return log_norm - target_score

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        #比如输入y_pred = (2,50,7),则K.shape(y_pred)[:-1] = (2,50)
        #这里为去除了最后一位数值7之后的形状
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        #这里的y_true = tf.Tensor([[0 0 0 ... 3 4 4 0 0]
        #[0 0 0...3 4 4 0 0]],shape = (2,50),相当于形状没有变化
        # 转为one hot
        y_true = K.one_hot(y_true, K.shape(self.trans)[0])
        loss = self.dense_loss(y_true, y_pred)
        return loss

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 逐标签取最大来粗略评测训练效果
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(isequal * mask) / K.sum(mask)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
        }
        base_config = super(ConditionalRandomField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class CRF(tf.keras.layers.Layer):
    """
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).
    Args:
        num_labels (int): the number of labels to tag each temporal input.
    Input shape:
        nD tensor with shape `(batch_size, sentence length, num_classes)`.
    Output shape:
        nD tensor with shape: `(batch_size, sentence length, num_classes)`.
    Masking
        This layer supports keras masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an embedding layer with the `mask_zero` parameter
        set to `True` or add a Masking Layer before this Layer
    """

    def __init__(self, sparse_target=True, **kwargs):
        self.transitions = None
        super(CRF, self).__init__(**kwargs)
        self.sparse_target = sparse_target
        self.sequence_lengths = None
        self.mask = None
        self.output_dim = None
    
    def get_config(self):
    #get_config这波没啥用，先不管它
        config = {
            "output_dim": self.output_dim,
            #self.output_dim = None
            "transitions": K.eval(self.transitions),
            #self.transitions = [input_shape[-1],input_shape[-1]]
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        # assert len(input_shape) == 3
        self.transitions = self.add_weight(
            name="transitions",
            shape=[self.output_dim, self.output_dim],
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs, mask=None, training=None):
        #print('inputs = ')
        #print(inputs)
        if mask is not None:
            self.sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
            self.mask = mask
        else:
            self.sequence_lengths = K.sum(K.ones_like(inputs[:, :, 0], dtype='int32'), axis=-1)
        #print('inputs[:,:,0] = ')
        #print(inputs[:,:,0])
        #inputs = (None,None,9),inputs[:,:,0] = (None,None)
        #print('K.ones_like(inputs[:,:,0] = ')
        #print(K.ones_like(inputs[:,:,0]))
        #K.ones_like(inputs[:,:,0]) = (None,None)
        #这点处理很巧妙，将原先的(batch_size,seq_len_size)全部用1填充满，
        #然后将最后一维的数值相加，得到的就是每一个长度构成的tensor
        #(data1,data2,...datan),每一个数值对应相应的长度
        if training:
            return inputs
        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, self.sequence_lengths
        )
        # tensorflow requires TRUE and FALSE branch has the same dtype
        return K.cast(viterbi_sequence, inputs.dtype)

    def loss(self, y_true, y_pred):
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)
        if len(y_pred.shape) == 2:
            y_pred = K.one_hot(K.cast(y_pred, 'int32'), self.output_dim)
        log_likelihood, _ = tfa.text.crf_log_likelihood(
            y_pred,
            y_true,
            self.sequence_lengths,
            transition_params=self.transitions,
        )
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.out_dim,)

    def compute_mask(self, inputs, mask=None):
        return mask

    # use crf decode to estimate accuracy
    def accuracy(self, y_true, y_pred):
        mask = self.mask
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)
        if len(y_pred.shape) == 3:
            y_pred, _ = tfa.text.crf_decode(
                y_pred, self.transitions, self.sequence_lengths
            )
        y_true = K.cast(y_true, y_pred.dtype)
        is_equal = K.equal(y_true, y_pred)
        is_equal = K.cast(is_equal, y_pred.dtype)
        if mask is None:
            return K.sum(is_equal) / K.sum(self.sequence_lengths)
        else:
            mask = K.cast(mask, y_pred.dtype)
            return K.sum(is_equal * mask) / K.sum(mask)