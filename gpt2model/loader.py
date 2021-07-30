from models import GPT2
import tensorflow as tf
import tensorflow.keras as keras
def _checkpoint_exists(ckpt_path):
    cktp_files = tf.io.gfile.glob(ckpt_path + "*")
    return len(cktp_files) > 0

def load_stock_weights(gpt2: GPT2, ckpt_path):
    #先构造完成bert模型之后再写这一部分，原因是构造Bert的模型过程中产生
    #模型的参数数值，然后再将从文件中读取出来的权重数值赋值到Bert之中去
    """
    Use this method to load the weights from a pre-trained gpt2 checkpoint into a gpt2 layer.

    :param bert: a BertModelLayer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    print('load_stock_weights')
    assert isinstance(gpt2, GPT2), "Expecting a BertModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(gpt2.weights) > 0, "GPT2Layer weights have not been instantiated yet. " \
    #                              "Please add the layer in a Keras model and call model.build() first!"
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
    #print('&&&stock_weights = &&&')
    #print(stock_weights)
    #print('&&&&&&')
    
    #get_variable_to_dtype_map().keys():只获取对应的变量名称
    gpt2_params = gpt2.weights
    param_values = keras.backend.batch_get_value(gpt2.weights)
    #之前使用param相当于重新定义了一个属性param属性，然后通过param属性去查找相应的
    #适配其他类型的权重实际上就是修改对应的参数名称，让它能够跟新的权重内容匹配上
    #print('param_values = ')
    #for data in param_values:
    #    print('data.shape = ')
    #    print(data.shape)
    gpt2model = []
    for  data  in  gpt2.weights:
         gpt2model.append(data.name)
    print('gpt2model = ')
    print(gpt2model)
    
    transformer_dicts = {
       'gpt2/embeddings/position_embeddings/embeddings:0':'gpt/embeddings/position_embeddings',
       'gpt2/embeddings/word_embeddings/embeddings:0':'gpt/embeddings/word_embeddings',
       'gpt2/layer_normalization/gamma:0':'gpt/output/LayerNorm/gamma',
       'gpt2/layer_normalization/beta:0':'gpt/output/LayerNorm/beta'
    }
    #字典内容：现有模型权重名称：原来模型的权重名称
    for layer_ndx in range(gpt2.num_layers):
        transformer_dicts.update({
            'gpt2/transformer_%d/attention/query/kernel:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/self/query/kernel'%(layer_ndx),
            'gpt2/transformer_%d/attention/query/bias:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/self/query/bias'%(layer_ndx),
            'gpt2/transformer_%d/attention/key/kernel:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/self/key/kernel'%(layer_ndx),
            'gpt2/transformer_%d/attention/key/bias:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/self/key/bias'%(layer_ndx),
            'gpt2/transformer_%d/attention/value/kernel:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/self/value/kernel'%(layer_ndx),
            'gpt2/transformer_%d/attention/value/bias:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/self/value/bias'%(layer_ndx),

            'gpt2/transformer_%d/dense0/kernel:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/output/dense/kernel'%(layer_ndx),
            'gpt2/transformer_%d/dense0/bias:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/output/dense/bias'%(layer_ndx),
            'gpt2/transformer_%d/layer_normalization_0/gamma:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/input/LayerNorm/gamma'%(layer_ndx),
            'gpt2/transformer_%d/layer_normalization_0/beta:0'%(layer_ndx):'gpt/transformer/layer_%d/attention/input/LayerNorm/beta'%(layer_ndx),
            'gpt2/transformer_%d/dense1/kernel:0'%(layer_ndx):'gpt/transformer/layer_%d/intermediate/dense/kernel'%(layer_ndx),
            'gpt2/transformer_%d/dense1/bias:0'%(layer_ndx):'gpt/transformer/layer_%d/intermediate/dense/bias'%(layer_ndx),
            'gpt2/transformer_%d/dense2/kernel:0'%(layer_ndx):'gpt/transformer/layer_%d/output/dense/kernel'%(layer_ndx),
            'gpt2/transformer_%d/dense2/bias:0'%(layer_ndx):'gpt/transformer/layer_%d/output/dense/bias'%(layer_ndx),
           
            'gpt2/transformer_%d/layer_normalization_1/gamma:0'%(layer_ndx):'gpt/transformer/layer_%d/input/LayerNorm/gamma'%(layer_ndx),
            'gpt2/transformer_%d/layer_normalization_1/beta:0'%(layer_ndx):'gpt/transformer/layer_%d/input/LayerNorm/beta'%(layer_ndx),
            
            'gpt2/mlm_dense/kernel:0':'gpt/embeddings/word_embeddings'
       })
    weight_value_tuples = []
    loaded_weights = set()
    skipped_weight_value_tuples = []
    skip_count = 0
    flags = 0
    for ndx, (param_value, param) in enumerate(zip(param_values,gpt2_params)):
        #param_value为对应的参数值
        #param为对应的参数
        stock_name = transformer_dicts[param.name]
        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)
            if param.name == 'gpt2/mlm_dense/kernel:0':
                ckpt_value = ckpt_value.transpose()
            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue
            weight_value_tuples.append((param, ckpt_value))
            #param对应的属性，包括名称以及参数，ckpt_value只有参数内容
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
        flags = flags+1
        
        if flags%20 == 0:
            keras.backend.batch_set_value(weight_value_tuples)
            weight_value_tuples = []
        
    #print('***weight_value_tuples = ***')
    #print(weight_value_tuples)
    keras.backend.batch_set_value(weight_value_tuples)
    #tf.keras.backend.batch_set_value(tuples):一次设置多个tensor变量的值
    #tuples:元组列表(tensor,value)。value应该是一个numpy数组。
    print("Done loading {} gpt2 weights from: {} into {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), ckpt_path, gpt2,skip_count, len(skipped_weight_value_tuples)))

    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)