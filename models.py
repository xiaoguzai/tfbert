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
                                           size_per_head = self.size_per_head,
                                           attention_probs_dropout_prob = 0.1,
                                           negative_infinity = -10000.0,
                                           intermediate_size = self.intermediate_size,
                                           mode = self.mode,
                                           solution = self.solution
                                          )
            encoder_layer.name = 'transformer_%d'%layer_ndx
            self.encoder_layer.append(encoder_layer)
        
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
        #当为list数组形式的时候可以手动传入对应的segment_ids
        shape = input_ids.shape
        #print('shape = ')
        #print(shape)
        maxlen = shape[1]
        batch_size = shape[0]
        #input_ids = tf.convert_to_tensor(input_ids)
        word_embeddings = self.word_embeddings_layer(input_ids)
        
        r"""
        assert_op = tf.compat.v2.debugging.assert_less_equal(maxlen,self.max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            position_embeddings = tf.slice(self.position_embeddings_table,
                                                [0,0],
                                                [maxlen,-1])
        """
        input_shape = K.shape(input_ids)
        batch_size,seq_len = input_shape[0],input_shape[1]
        position_ids = K.arange(0,seq_len,dtype='int32')[None]
        position_embeddings = self.position_embeddings_table[None,:seq_len]
        #取出对应的position_embeddings内容
        #position_embeddings = Tensor("bert/embeddings/strided_slice_3:0",shape=(1,None,768),
        #dtype=float32)
        #!!!不能使用None作为长度的原因!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #上面的tf.slice会报错
        
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
        
        results = word_embeddings+position_embeddings
        #results = tf.reshape(position_embeddings,currentshape)
        
        
        #word_embeddings+tf.reshape(position_embeddings,currentshape)
        #这里报错:required broadcastable shapes at loc(unknown)
        results = results+segment_embeddings
        #results = results+segment_embeddings
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
    def build(self,input_shape):
        self.attention = AttentionLayer(initializer_range = self.initializer_range,
                                        num_attention_heads = self.num_attention_heads,
                                        size_per_head = self.size_per_head,
                                        query_activation = self.query_activation,
                                        key_activation = self.key_activation,
                                        value_activation = self.value_activation,
                                        attention_probs_dropout_prob = self.attention_probs_dropout_prob,
                                        negative_infinity = self.negative_infinity,
                                        name = "attention",
                                        mode = self.mode,
                                        solution = self.solution)
        self.dense0 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "dense0")
        self.dropout0 = keras.layers.Dropout(rate=self.hidden_dropout)
        self.layer_norm0 = LayerNormalization()
        self.layer_norm0.name = "layer_normalization_0"
        #自己定义的keras.layers层定义名称之后，名称赋值并不能赋值上
        self.dense = keras.layers.Dense(units = self.intermediate_size,
                                        kernel_initializer = self.create_initializer(),
                                        activation = self.get_activation('gelu'),
                                        name = "dense")
        #错误点1：这里的activation少加了一个激活的gelu函数
        self.dense1 = keras.layers.Dense(units = self.embedding_size,
                                        kernel_initializer = self.create_initializer(),
                                        name = "dense1")
        self.dropout1 = keras.layers.Dropout(rate=self.hidden_dropout)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm1.name = "layer_normalization_1"
        super(Transformer,self).build(input_shape)
                                        
    def call(self,inputs,**kwargs):
        #print('Transformer call')
        residual = inputs[0]
        embedding_output = self.attention(inputs)
        #transformer结构图中确实是residual inputs在attention之前
        #然而这里进行分类任务的时候实测residual inputs在attention之后效果更好
        #residual = inputs
        embedding_output = self.dense0(embedding_output)
        #print('after dense')
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
        embedding_output = self.dense1(embedding_output)
        embedding_output = self.dropout1(embedding_output)
        embedding_output = self.layer_norm1(residual+embedding_output)
        return [embedding_output,inputs[1]]

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
    
    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
    
    def build(self,input_shape):
        if isinstance(input_shape,list):
            assert 2 == len(input_shape),"Expecting inputs to be a [input_ids,token_type_ids] list"
        dense_units = self.num_attention_heads*self.size_per_head
        #768=12*64,12头，每一头64个维度
        self.query_layer = keras.layers.Dense(units=dense_units,activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer = keras.layers.Dense(units=dense_units,activation=self.key_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="key")
        self.value_layer = keras.layers.Dense(units=dense_units,activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.attention_probs_dropout_prob)
        super(AttentionLayer,self).build(input_shape)

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
    
    def call(self,inputs,**kwargs):
       #!!!这里报错:Layer attention expects 1 input(s),but it received 2 input tensors
       #!!!遥想到上面的Embedding放了两个Tensor却没有被报错,所以这段可以仿照上面的Embedding对应
       #!!!层进行书写相应内容
       #!!!再回看一下bert4keras如何实现的放入一个数组多个值的操作
        input_ids,segment_ids = inputs[0],inputs[1]
        input_shape = tf.shape(input=inputs[0])
        batch_size,seq_len,width = input_shape[0],input_shape[1],input_shape[2]
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
        query = self.query_layer(inputs[0])
        #query = (None,128,768)
        key = self.key_layer(inputs[0])
        value = self.value_layer(inputs[0])
        #query = (None,128,768)
        query = transpose_for_scores(query,seq_len)
        key = transpose_for_scores(key,seq_len)
        #(None,12,128,64),batch_size = 128
        #query = (None,12,None,64),key = (None,12,None,64)
        attention_scores = tf.matmul(query,key,transpose_b=True)
        #(None,12,128,64)*(None,12,64,128) = (None,12,128,128)
        #实现将批次在矩阵之中提取出来的操作
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
        #(5,12,128,128)*(5,12,128,64) = (5,12,128,64)
        context_layer = tf.transpose(a=context_layer,perm=[0,2,1,3])
        #(None,64,12,128)
        #context_layer = (None,12,None,64)
        output_shape = [batch_size,seq_len,
                       self.num_attention_heads*self.size_per_head]
        #output_shape = [None,None,12,64]
        context_layer = tf.reshape(context_layer,output_shape)
        #最终转为context_layer = (5,128,12,64)
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