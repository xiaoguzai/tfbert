from models import Bert
import tensorflow as tf
import tensorflow.keras as keras
import torch
def _checkpoint_exists(ckpt_path):
    cktp_files = tf.io.gfile.glob(ckpt_path + "*")
    return len(cktp_files) > 0

def load_stock_weights(bert: Bert, ckpt_path):
    #先构造完成bert模型之后再写这一部分，原因是构造Bert的模型过程中产生
    #模型的参数数值，然后再将从文件中读取出来的权重数值赋值到Bert之中去
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param bert: a BertModelLayer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    assert isinstance(bert, Bert), "Expecting a BertModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(bert.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
    #                              "Please add the layer in a Keras model and call model.build() first!"
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
    #stock_weights为从文件中读取的对应关键值名称
    bert_params = bert.weights
    bertmodelname = []
    for data in bert.weights:
        bertmodelname.append(data.name)
    param_values = keras.backend.batch_get_value(bert.weights)
    #之前使用param相当于重新定义了一个属性param属性，然后通过param属性去查找相应的
    #适配其他类型的权重实际上就是修改对应的参数名称，让它能够跟新的权重内容匹配上
    transformer_dicts = {
       'bert/embeddings/position_embeddings/embeddings:0':'bert/embeddings/position_embeddings',
       'bert/embeddings/word_embeddings/embeddings:0':'bert/embeddings/word_embeddings',
       'bert/embeddings/segment_embeddings/embeddings:0':'bert/embeddings/token_type_embeddings',
       'bert/embeddings/layer_normalization/gamma:0':'bert/embeddings/LayerNorm/gamma',
       'bert/embeddings/layer_normalization/beta:0':'bert/embeddings/LayerNorm/beta',
        
       'bert/mlm_dense0/kernel:0':'cls/predictions/transform/dense/kernel',
       'bert/mlm_dense0/bias:0':'cls/predictions/transform/dense/bias',
       'bert/mlm_dense1/kernel:0':'bert/embeddings/word_embeddings',
       'bert/mlm_dense1/bias:0':'cls/predictions/output_bias',
       'bert/mlm_norm/gamma:0':'cls/predictions/transform/LayerNorm/gamma',
       'bert/mlm_norm/beta:0':'cls/predictions/transform/LayerNorm/beta'
    }
    for layer_ndx in range(bert.num_layers):
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
    weight_value_tuples = []
    loaded_weights = set()
    skipped_weight_value_tuples = []
    skip_count = 0
    for ndx, (param_value, param) in enumerate(zip(param_values,bert_params)):
        #param_value为对应的参数值
        #param为对应的参数
        stock_name = transformer_dicts[param.name]
        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)
            if param.name == 'bert/mlm_dense1/kernel:0':
                ckpt_value = ckpt_value.transpose()
            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue
            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)
    #tf.keras.backend.batch_set_value(tuples):一次设置多个tensor变量的值
    #tuples:元组列表(tensor,value)。value应该是一个numpy数组。
    print("Done loading {} BERT weights from: {} into {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), ckpt_path, bert,skip_count, len(skipped_weight_value_tuples)))

    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)
            
def load_pytorch_weight(bert,resolved_archive_file):
    state_dict = None
    print('load_pytorch_weight111')
    if state_dict is None:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
            #print('state_dict = ')
            #print(state_dict.keys())
            #print('model state_dict = ')
            #print(model.state_dict())
            file_name = list(state_dict.keys())
        except Exception:
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                f"at '{resolved_archive_file}'"
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )
    transformer_dicts = {
       'bert/embeddings/position_embeddings/embeddings:0':'module.bertembeddings.position_embeddings_layer.weight',
       'bert/embeddings/word_embeddings/embeddings:0':'module.bertembeddings.word_embeddings_layer.weight',
       'bert/embeddings/segment_embeddings/embeddings:0':'module.bertembeddings.segment_embeddings_layer.weight',
       'bert/embeddings/layer_normalization/gamma:0':'module.bertembeddings.layer_normalization.weight',
       'bert/embeddings/layer_normalization/beta:0':'module.bertembeddings.layer_normalization.bias',
        
       'bert/mlm_dense0/kernel:0':'module.mlm_dense0.weight',
       'bert/mlm_dense0/bias:0':'module.mlm_dense0.bias',
       'bert/mlm_dense1/kernel:0':'module.mlm_dense1.weight',
       'bert/mlm_dense1/bias:0':'module.mlm_dense1.bias',
       'bert/mlm_norm/gamma:0':'module.mlm_norm.weight',
       'bert/mlm_norm/beta:0':'module.mlm_norm.bias',
    }
    bert_params = bert.weights
    model_name = []
    for data in bert.weights:
        model_name.append(data.name)
    param_values = keras.backend.batch_get_value(bert.weights)
    for layer_ndx in range(bert.num_layers):
        transformer_dicts.update({
            'bert/transformer_%d/attention/query/kernel:0'%(layer_ndx):'module.bert_encoder_layer.%d.attention.query_layer.weight'%(layer_ndx),
            #注意中间有冒号，两边要分开进行赋值
            'bert/transformer_%d/attention/query/bias:0'%(layer_ndx):'module.bert_encoder_layer.%d.attention.query_layer.bias'%(layer_ndx),
            'bert/transformer_%d/attention/key/kernel:0'%(layer_ndx):'module.bert_encoder_layer.%d.attention.key_layer.weight'%(layer_ndx),
            'bert/transformer_%d/attention/key/bias:0'%(layer_ndx):'module.bert_encoder_layer.%d.attention.key_layer.bias'%(layer_ndx),
            'bert/transformer_%d/attention/value/kernel:0'%(layer_ndx):'module.bert_encoder_layer.%d.attention.value_layer.weight'%(layer_ndx),
            'bert/transformer_%d/attention/value/bias:0'%(layer_ndx):'module.bert_encoder_layer.%d.attention.value_layer.bias'%(layer_ndx),
            
            'bert/transformer_%d/dense0/kernel:0'%(layer_ndx):'module.bert_encoder_layer.%d.dense0.weight'%(layer_ndx),
            'bert/transformer_%d/dense0/bias:0'%(layer_ndx):'module.bert_encoder_layer.%d.dense0.bias'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_0/gamma:0'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm0.weight'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_0/beta:0'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm0.bias'%(layer_ndx),
            
            'bert/transformer_%d/dense/kernel:0'%(layer_ndx):'module.bert_encoder_layer.%d.dense.weight'%(layer_ndx),
            'bert/transformer_%d/dense/bias:0'%(layer_ndx):'module.bert_encoder_layer.%d.dense.bias'%(layer_ndx),

            'bert/transformer_%d/dense1/kernel:0'%(layer_ndx):'module.bert_encoder_layer.%d.dense1.weight'%(layer_ndx),
            'bert/transformer_%d/dense1/bias:0'%(layer_ndx):'module.bert_encoder_layer.%d.dense1.bias'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_1/gamma:0'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm1.weight'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_1/beta:0'%(layer_ndx):'module.bert_encoder_layer.%d.layer_norm1.bias'%(layer_ndx),
            
        })
    weight_value_tuples = []
    skipped_weight_value_tuples = []
    skip_count = 0
    loaded_weights = []
    used_name = []
    for ndx, (param_value,param) in enumerate(zip(param_values,bert_params)):
        if param.name in transformer_dicts:
            stock_name = transformer_dicts[param.name]
            stock_value = state_dict[stock_name]
            stock_value = stock_value.numpy()
            if stock_name == 'module.bertembeddings.word_embeddings_layer.weight':
                stock_value = stock_value[:param_value.shape[0]]
            if param.name == 'bert/mlm_dense1/bias:0':
                stock_value = stock_value[:param_value.shape[0]]
            #if param.name == 'bert/mlm_dense1/kernel:0':
            if 'kernel:0' in param.name:
                stock_value = stock_value.transpose()
            #!!!pytorch权重为原先线性网络层权重的转置
            if param_value.shape != stock_value.shape:
                print('Unused weight')
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param_value.shape,
                                                                 stock_name, stock_value.shape))
                skipped_weight_value_tuples.append((param.name,stock_value))
                continue
            weight_value_tuples.append((param,stock_value))
            used_name.append(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param_name, stock_name, resolved_archive_file))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)
    print("Done loading {} NEZHA weights from: {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), resolved_archive_file,skip_count, len(skipped_weight_value_tuples)))

    #print("Unused weights from checkpoint:",
    #      "\n\t" + "\n\t".join(sorted(file_name.difference(used_name))))
    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(set(file_name).difference(set(used_name))))
    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)
