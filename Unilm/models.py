import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import os
import numpy as np
#import tensorflow.keras.layers.Layer as Layer
#No module named 'tensorflow.keras.layers.Layer'
#之前的写法是import tensorflow.keras.layers.Layer as Layer报错
class  GPT2(tf.keras.layers.Layer):
    name = 'gpt2'
    def __init__(self,
                maxlen = 128,#0
                initializer_range=0.02,#1
                max_position_embeddings=1024,#2
                embedding_size=2560,#4
                project_embeddings_with_bias=True,#5
                vocab_size=30000,#6
                hidden_dropout=0.1,#10
                extra_tokens_vocab_size=None,#11
                project_position_embeddings=True,#12
                mask_zero=True,#13
                adapter_size=None,#14
                hidden_act='gelu',#15
                adapter_init_scale=0.001,#16
                num_attention_heads=32,#17
                size_per_head=None,#18
                attention_probs_dropout_prob=0.1,#22
                negative_infinity=-10000.0,#23
                intermediate_size=10240,#24
                intermediate_activation='gelu',#25
                num_layers=32,#26
                batch_size = 256,
                #获取对应的切分分割字符内容
                mlm = False,
                mlm_activation = 'softmax',
                *args, **kwargs):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        super(GPT2, self).__init__()
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
        self.project_position_embeddingFs = project_position_embeddings#12
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
        self.mlm = mlm
        self.mlm_activation = mlm_activation

    def build(self, input_ids):
        #加一个是否len为2的判断
        self.batch_size = input_ids[0]
        self.maxlen = input_ids[1]
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
                                           size_per_head = self.size_per_head,
                                           attention_probs_dropout_prob = 0.1,
                                           negative_infinity = -10000.0,
                                           intermediate_size = self.intermediate_size,
                                           index = layer_ndx
                                          )
            encoder_layer.name = 'transformer_%d'%layer_ndx
            self.encoder_layer.append(encoder_layer)
            #这里使用add_weight作为bias进行操作
            #self.mlm_activation_layer = keras.layers.Activation(self.mlm_activation)
        self.layer_normalization = LayerNormalization()
        self.layer_normalization.name = 'layer_normalization'
        self.dropout = keras.layers.Dropout(rate=self.hidden_dropout)
        if self.mlm:
            self.mlm_dense = keras.layers.Dense(units = self.vocab_size,
                                          kernel_initializer = self.create_initializer(),
                                          use_bias = False,
                                          name = "mlm_dense")
        super(GPT2, self).build(input_ids)
    
    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
    
    def call(self,input_ids):
        #input_ids = (?,128)
        input_ids = tf.convert_to_tensor(input_ids)
        output_embeddings = self.embeddings(input_ids)
        #print('output_embeddings = ')
        #print(output_embeddings)
        #output_embeddings = (?,128,2560)
        #这里定义为多个transformer的内容
        #output_embeddings = [1,7,768]
        #到output_embeddings部分输出相同
        
        for encoder_layer in self.encoder_layer:
            output_embeddings = encoder_layer(output_embeddings)
        output_embeddings = self.layer_normalization(output_embeddings)
        print('output_embeddings1 = ')
        print(output_embeddings)
        output_embeddings = self.dropout(output_embeddings)
        print('output_embeddings2 = ')
        print(output_embeddings)
        output_embeddings = self.mlm_dense(output_embeddings)
        print('output_embeddings3 = ')
        print(output_embeddings)
        if self.mlm_activation == 'softmax':
            print('run softmax')
            output_embeddings = keras.activations.softmax(output_embeddings,axis=-1)
        #之前这里的output_embeddings写成了output_embeddigns!!!啊啊啊!!!
        #outputs = output_embeddings
        print('output_embeddings4 = ')
        print(output_embeddings)
        outputs = output_embeddings
        print('transformer return outputs')
        print('outputs = ')
        print(outputs)
        return outputs
    
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

class  Embeddings(tf.keras.layers.Layer):
    #实现word_embedding,segment_embedding,position_embedding的相加求和结果
    name = 'embeddings'
    def __init__(self,
                 vocab_size = 30000,
                 embedding_size = 2560,
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
        
    #def compute_mask(self,inputs,mask=None):
        #Embedding之中定义compute_mask函数的选项，这里输出的mask会继续到下一层
        #mask = K.not_equal(inputs,0)
        r"""print('mask = ')
        print(mask)
        mask1 = K.ones_like(mask[:,:1],dtype='bool')
        print('mask1 = ')
        print(mask1)
        mask2 = mask[:,1:]
        print('mask2 = ')
        print(mask2)
        """
        #return mask
        
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
        #gpt2的特点：去除了相加的部分以及LayerNormalization
        super(Embeddings,self).build(inputs)
    def call(self,input_ids):
        #assert len(input_ids) > 0,"Bert input length cannot be zero.Please adjust your input"
        #inputs = (None,128)
        #这种情形能够产生(None,128)的对应Tensor
        shape = input_ids.shape
        #print('shape = ')
        #print(shape)
        maxlen = shape[1]
        batch_size = shape[0]
        #input_ids = tf.convert_to_tensor(input_ids)
        word_embeddings = self.word_embeddings_layer(input_ids)
        assert_op = tf.compat.v2.debugging.assert_less_equal(maxlen,self.max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            position_embeddings = tf.slice(self.position_embeddings_table,
                                                [0,0],
                                                [maxlen,-1])
        currentshape = [1,maxlen,self.embedding_size]
        results = word_embeddings+tf.reshape(position_embeddings,currentshape)
        return results

class  Transformer(tf.keras.layers.Layer):
    name = 'transformer'
    def __init__(self,
                 initializer_range = 0.02,
                 embedding_size = 2560,
                 hidden_dropout = 0.1,
                 adapter_size = None,
                 hidden_act = 'gelu',
                 adapter_init_scale = 0.001,
                 num_attention_heads = 32,
                 size_per_head = 80,
                 query_activation = None,
                 key_activation = None,
                 value_activation = None,
                 attention_probs_dropout_prob = 0.1,
                 negative_infinity = -10000.0,
                 intermediate_size = 10240,
                 index = 0,
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
        self.index = index
    def build(self,input_shape):
        self.layer_normalization0 = LayerNormalization()
        self.layer_normalization0.name = "layer_normalization_0"
        self.attention = MultiMaskAttentionLayer(initializer_range = self.initializer_range,
                                        num_attention_heads = self.num_attention_heads,
                                        size_per_head = self.size_per_head,
                                        query_activation = self.query_activation,
                                        key_activation = self.key_activation,
                                        value_activation = self.value_activation,
                                        attention_probs_dropout_prob = self.attention_probs_dropout_prob,
                                        negative_infinity = self.negative_infinity,
                                        name = "attention",
                                        index = self.index)
        self.dense0 = keras.layers.Dense(units = self.embedding_size,
                                         kernel_initializer = self.create_initializer(),
                                         name = "dense0")
        self.dropout_layer0 = keras.layers.Dropout(rate=self.hidden_dropout)
        self.layer_normalization1 = LayerNormalization()
        self.layer_normalization1.name = "layer_normalization_1"
        self.dense1 = keras.layers.Dense(units = self.intermediate_size,
                                         kernel_initializer = self.create_initializer(),
                                         activation = self.get_activation('gelu'),
                                         name = "dense1")
        #!!!这里使用gelu激活函数之后与结果类似
        self.dense2 = keras.layers.Dense(units = self.embedding_size,
                                         kernel_initializer = self.create_initializer(),
                                         name = "dense2")
        r"""self.dense2 = keras.layers.Dense(units = self.embedding_size,
                                         kernel_initializer = self.create_initializer(),
                                         activation = self.get_activation('gelu'),
                                         name = "dense2")
        """
        self.dropout_layer1 = keras.layers.Dropout(rate=self.hidden_dropout)

        super(Transformer,self).build(input_shape)
                                         
    def call(self,inputs,**kwargs):
        residual = inputs
        embedding_output = self.layer_normalization0(inputs)
        embedding_output = self.attention(embedding_output)
        embedding_output = self.dense0(embedding_output)
        embedding_output = self.dropout_layer0(embedding_output)
        currents = residual+embedding_output
        embedding_output = self.layer_normalization1(currents)
        embedding_output = self.dense1(embedding_output)
        embedding_output = self.dense2(embedding_output)
        #两个dense形成forward层
        embedding_output = self.dropout_layer1(embedding_output)
        return currents+embedding_output
        
        #return embedding_output

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)

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

class MultiMaskAttentionLayer(tf.keras.layers.Layer):
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
                 index = 0,
                 **kwargs):
        super(MultiMaskAttentionLayer,self).__init__()
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
        self.index = index

    def compute_attention_bias(self,inputs):
        #inputs = (None,24,2560)
        seq_len = tf.shape(input=inputs)
        seq_len = seq_len[2]
        idxs = K.arange(0,seq_len)
        #在seq_len = 24的情况下，idxs = tf.Tensor([0,1,2...23],shape=(24,),dtype=int32)
        mask = idxs[None,:] <= idxs[:,None]
        r"""
        mask = tf.Tensor(
        [[True False False]
         [True  True False]
         [True  True  True]],shape=(24,24),dtype=bool)
         形成下三角对应的tensor数组
        """
        #mask = (None,None)
        mask = K.cast(mask,K.floatx())
        results = -(1-mask[None,None])*1e12
        r"""results = (1,1,24,24),tf.Tensor([[[[-0.e+00 -1.e+12 -1.e+12...]]
        [-0.e+00 -0.e+00 -1.e+12...]]])
        """
        return results
        r"""
        这里的mask[None,None]将mask的形状扩大维度，
        本身的形状为(3,3),现在的形状为(1,1,3,3)
        输出 = array([[[[-0.e+00,-1.e+12,-1.e+12],
                       [-0.e+00,-0.e+00,-1.e+12],
                       [-0.e+00,-0.e+00,-0.e+00]]]],dtype=float32)
        """

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
    
    def build(self,input_shape):
        #build之中可以通过input_spec来定义对应新的keras.layers.InputSpec
        dense_units = self.num_attention_heads*self.size_per_head
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
        self.dropout_layer = keras.layers.Dropout(self.attention_probs_dropout_prob)
        self.output_layer = keras.layers.Dense(units=dense_units,activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="output")
        super(MultiMaskAttentionLayer,self).build(input_shape)

    def compute_output_shape(self,input_shape):
        from_shape = input_shape
        output_shape = [from_shape[0],from_shape[1],self.num_attention_heads*self.size_per_head]
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        return output_shape
        
    def call(self,inputs,mask=None,**kwargs):
        #inputs = (?,128,2560)
        #bias_data = self.compute_attention_bias(inputs)
        #bias_data = (1,1,?,?)
        input_shape = tf.shape(input=inputs)
        #或者shape = K.int_shape(inputs)
        #print('input_shape = ')
        #print(input_shape)
        batch_size,seq_len,width = input_shape[0],input_shape[1],input_shape[2]
        def transpose_for_scores(input_tensor,seq_len):
        #改变数组形状，由[batch_size,from_seq_len,num_attention_heads*size_per_head]
        #->[batch_size,num_attention_heads,from_seq_len,size_per_head]
            output_shape = [batch_size,seq_len,
                           self.num_attention_heads,self.size_per_head]
            #output_shape = tf.convert_to_tensor(output_shape)
            #output_shape = np.array(output_shape)
            #input_tensor = ("bert/transformer/attention_layer/query/
            #BiasAdd:0",shape=(None,128,768))
            #output_shape = [None,128,12,64]
            #output_shape = [tf.Tensor,shape=(),tf.Tensor,shape=(),12,64]
            output_tensor = K.reshape(input_tensor,output_shape)
            #小函数能够使用外面的变量
            #注意这里的output_shape必须放入tensor类型的数值
            #return tf.transpose(a=output_tensor,perm=[0,2,1,3])
            return output_tensor
        q_mask = mask
        v_mask = mask
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        query = transpose_for_scores(query,seq_len)
        key = transpose_for_scores(key,seq_len)
        value = transpose_for_scores(value,seq_len)
        #query = (None,None,32,80),key = (None,None,32,80),value = (None,None,32,80)
        r"""
        这里的mask[None,None]将mask的形状扩大维度，
        本身的形状为(3,3),现在的形状为(1,1,3,3)
        输出 = array([[[[-0.e+00,-1.e+12,-1.e+12],
                       [-0.e+00,-0.e+00,-1.e+12],
                       [-0.e+00,-0.e+00,-0.e+00]]]],dtype=float32)
        注意!!!-1.e+12代表1*10^12，1后面带12个0，是一万亿的意思，
        这里使用-1.e+12加上query之后再进行softmax，后面的对应值在softmax
        之后就会变为0
        """
        #query = (None,None,32,80)
        query = tf.transpose(a=query,perm=[0,2,1,3])
        #query = (None,32,None,80)
        key = tf.transpose(a=key,perm=[0,2,3,1])
        #key = (None,32,80,None)
        #query = (None,32,None,80),key = (None,32,80,None),value = (None,None,32,80)
        attention_scores = tf.matmul(query,key)
        #attention_scores = (None,32,None,None)
        attention_scores = attention_scores/tf.sqrt(float(self.size_per_head))
        bias_data = self.compute_attention_bias(attention_scores)
        #attention_scores = [1,32,24,24]
        attention_scores = attention_scores+bias_data
        #这里实现query*key/根号(dmodel)
        #这里采用先加和再掩码的方式，attention_scores = [1,32,24,24]
        #bias_data = [1,1,24,24],并且bias_data的右上三角全部是采用无穷大的操作
        #此时使用attention_scores+bias_data让右上三角部分的对应值变成无穷大的内容
        values = -1e12
        #4.attention_scores = attention_scores*v_mask+values*(1-v_mask)
        #原理：v_mask的内容为是否不为零，当不为零的时候v_mask为1(True)，此时1-v_mask为0(False)
        #后面的values*(1-v_mask)=0，values为负无穷就不会影响到对应的results的相应值
        #而如果v_mask为0的时候后面的1-v_mask的对应值为1，此时后面加上的对应值为负无穷
        #就会将当前的这个位置标记为负无穷，softmax之后这个对应的位置就为零，这相当于另外一种对应的
        #mask的标记操作
        #attention_scores = (None,32,None,None)
        attention_probs = tf.nn.softmax(attention_scores)
        #attention_probs = (None,32,None,80)
        value = tf.transpose(a=value,perm=[0,2,1,3])
        #value = (None,32,None,80)
        context_layer = tf.matmul(attention_probs,value)
        #results = (None,32,None,80)
        context_layer = tf.transpose(a=context_layer,perm=[0,2,1,3])
        output_shape = [batch_size,seq_len,
                        self.num_attention_heads*self.size_per_head]
        context_layer = tf.reshape(context_layer,output_shape)
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
