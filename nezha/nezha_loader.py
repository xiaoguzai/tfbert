
import tensorflow as tf
import tensorflow.keras as keras
#from nezha import Nezha
from nezha import Bert
import h5py
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.saving import hdf5_format

def _checkpoint_exists(ckpt_path):
    cktp_files = tf.io.gfile.glob(ckpt_path + "*")
    return len(cktp_files) > 0

def load_nezha_stock_weights(nezha: Bert, ckpt_path):
    #先构造完成bert模型之后再写这一部分，原因是构造Bert的模型过程中产生
    #模型的参数数值，然后再将从文件中读取出来的权重数值赋值到Bert之中去
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param bert: a BertModelLayer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    print('load_nezha_stock_weights')
    assert isinstance(nezha,Bert), "Expecting a NezhaModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(nezha.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
    #                              "Please add the layer in a Keras model and call model.build() first!"
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
    #stock_weights为从文件中读取的对应关键值名称
    nezha_params = nezha.weights
    nezhamodelname = []
    for data in nezha.weights:
        nezhamodelname.append(data.name)
    #这里读取出来的nezhamodelname为我定义的权重中的名称
    param_values = keras.backend.batch_get_value(nezha.weights)
    #之前使用param相当于重新定义了一个属性param属性，然后通过param属性去查找相应的
    #适配其他类型的权重实际上就是修改对应的参数名称，让它能够跟新的权重内容匹配上
    transformer_dicts = {
       'nezha/embeddings/word_embeddings/embeddings:0':'bert/embeddings/word_embeddings',
       'nezha/embeddings/segment_embeddings/embeddings:0':'bert/embeddings/token_type_embeddings',
       'nezha/embeddings/layer_normalization/gamma:0':'bert/embeddings/LayerNorm/gamma',
       'nezha/embeddings/layer_normalization/beta:0':'bert/embeddings/LayerNorm/beta',
    }
    for layer_ndx in range(nezha.num_layers):
        transformer_dicts.update({
            'nezha/transformer_%d/attention/query/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/query/kernel'%(layer_ndx),
            #注意中间有冒号，两边要分开进行赋值
            'nezha/transformer_%d/attention/query/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/query/bias'%(layer_ndx),
            'nezha/transformer_%d/attention/key/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/key/kernel'%(layer_ndx),
            'nezha/transformer_%d/attention/key/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/key/bias'%(layer_ndx),
            'nezha/transformer_%d/attention/value/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/value/kernel'%(layer_ndx),
            'nezha/transformer_%d/attention/value/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/self/value/bias'%(layer_ndx),
            
            'nezha/transformer_%d/dense0/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/dense/kernel'%(layer_ndx),
            'nezha/transformer_%d/dense0/bias:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/dense/bias'%(layer_ndx),
            'nezha/transformer_%d/layer_normalization_0/gamma:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/LayerNorm/gamma'%(layer_ndx),
            'nezha/transformer_%d/layer_normalization_0/beta:0'%(layer_ndx):'bert/encoder/layer_%d/attention/output/LayerNorm/beta'%(layer_ndx),
            
            'nezha/transformer_%d/dense/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/intermediate/dense/kernel'%(layer_ndx),
            'nezha/transformer_%d/dense/bias:0'%(layer_ndx):'bert/encoder/layer_%d/intermediate/dense/bias'%(layer_ndx),

            'nezha/transformer_%d/dense1/kernel:0'%(layer_ndx):'bert/encoder/layer_%d/output/dense/kernel'%(layer_ndx),
            'nezha/transformer_%d/dense1/bias:0'%(layer_ndx):'bert/encoder/layer_%d/output/dense/bias'%(layer_ndx),
            'nezha/transformer_%d/layer_normalization_1/gamma:0'%(layer_ndx):'bert/encoder/layer_%d/output/LayerNorm/gamma'%(layer_ndx),
            'nezha/transformer_%d/layer_normalization_1/beta:0'%(layer_ndx):'bert/encoder/layer_%d/output/LayerNorm/beta'%(layer_ndx),
            
            'nezha/mlm_dense0/kernel:0':'cls/predictions/transform/dense/kernel',
            'nezha/mlm_dense0/bias:0':'cls/predictions/transform/dense/bias',
            'nezha/mlm_dense1/kernel:0':'bert/embeddings/word_embeddings',
            'nezha/mlm_dense1/bias:0':'cls/predictions/output_bias',
            'nezha/mlm_norm/gamma:0':'cls/predictions/transform/LayerNorm/gamma',
            'nezha/mlm_norm/beta:0':'cls/predictions/transform/LayerNorm/beta',
            'nezha/pooler/kernel:0':'bert/pooler/dense/kernel',
            'nezha/pooler/bias:0':'bert/pooler/dense/bias'
        })
    #左边为我定义的模型之中的名称，右边为读取的权重之中的内容
    weight_value_tuples = []
    loaded_weights = set()
    skipped_weight_value_tuples = []
    skip_count = 0
    for ndx, (param_value, param) in enumerate(zip(param_values,nezha_params)):
        #param_value为对应的参数值,先从自己模型中取出的param_values中得到参数
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
    print("Done loading {} NEZHA weights from: {} into {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), ckpt_path, nezha,skip_count, len(skipped_weight_value_tuples)))

    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)

def load_nezha_tf_weights_from_h5(nezha:Bert, resolved_archive_file, _prefix=None):
    #resolved_archive_file为对应的路径
    f = h5py.File('/home/xiaoguzai/下载/transformer-bert-base-uncased/tf_model.h5', 'r')
    stock_names = set()
    for root_name, g in f.items():
        #读取主目录文件，主目录有'bert','nsp__cls','mlm__cls'
        for _, weights_dirs in g.attrs.items():
            #g为存储的属性文件内容
            for i in weights_dirs:
                name = root_name + "/" + str(i, encoding="utf-8")
                data = f[name]
                stock_names.add(data.name[1:])
    #stock_weights用来保存为文件中所有的权重值
    nezha_params = nezha.weights
    nezhamodelname = []
    for data in nezha.weights:
        nezhamodelname.append(data.name)
    param_values = keras.backend.batch_get_value(nezha.weights)
    transformer_dicts = {
       'bert/embeddings/position_embeddings/embeddings:0':'bert/tf_bert_for_pre_training/bert/embeddings/position_embeddings/embeddings:0',
       'bert/embeddings/word_embeddings/embeddings:0':'bert/tf_bert_for_pre_training/bert/embeddings/word_embeddings/weight:0',
       'bert/embeddings/segment_embeddings/embeddings:0':'bert/tf_bert_for_pre_training/bert/embeddings/token_type_embeddings/embeddings:0',
       'bert/embeddings/layer_normalization/gamma:0':'bert/tf_bert_for_pre_training/bert/embeddings/LayerNorm/gamma:0',
       'bert/embeddings/layer_normalization/beta:0':'bert/tf_bert_for_pre_training/bert/embeddings/LayerNorm/beta:0',
       
       'bert/mlm_dense0/kernel:0':'mlm___cls/tf_bert_for_pre_training/mlm___cls/predictions/transform/dense/kernel:0',
       'bert/mlm_dense0/bias:0':'mlm___cls/tf_bert_for_pre_training/mlm___cls/predictions/transform/dense/bias:0',
       'bert/mlm_dense1/kernel:0':'mlm___cls/tf_bert_for_pre_training/bert/embeddings/word_embeddings/weight:0',
       'bert/mlm_dense1/bias:0':'mlm___cls/tf_bert_for_pre_training/mlm___cls/predictions/bias:0',
       'bert/mlm_norm/gamma:0':'mlm___cls/tf_bert_for_pre_training/mlm___cls/predictions/transform/LayerNorm/gamma:0',
       'bert/mlm_norm/beta:0':'mlm___cls/tf_bert_for_pre_training/mlm___cls/predictions/transform/LayerNorm/beta:0'
    }
    for layer_ndx in range(nezha.num_layers):
        transformer_dicts.update({
            'bert/transformer_%d/attention/query/kernel:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/self/query/kernel:0'%(layer_ndx),
            #注意中间有冒号，两边要分开进行赋值
            'bert/transformer_%d/attention/query/bias:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/self/query/bias:0'%(layer_ndx),
            'bert/transformer_%d/attention/key/kernel:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/self/key/kernel:0'%(layer_ndx),
            'bert/transformer_%d/attention/key/bias:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/self/key/bias:0'%(layer_ndx),
            'bert/transformer_%d/attention/value/kernel:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/self/value/kernel:0'%(layer_ndx),
            'bert/transformer_%d/attention/value/bias:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/self/value/bias:0'%(layer_ndx),
            
            'bert/transformer_%d/dense0/kernel:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/output/dense/kernel:0'%(layer_ndx),
            'bert/transformer_%d/dense0/bias:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/output/dense/bias:0'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_0/gamma:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/output/LayerNorm/gamma:0'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_0/beta:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/attention/output/LayerNorm/beta:0'%(layer_ndx),
            
            'bert/transformer_%d/dense/kernel:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/intermediate/dense/kernel:0'%(layer_ndx),
            'bert/transformer_%d/dense/bias:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/intermediate/dense/bias:0'%(layer_ndx),

            'bert/transformer_%d/dense1/kernel:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/output/dense/kernel:0'%(layer_ndx),
            'bert/transformer_%d/dense1/bias:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/output/dense/bias:0'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_1/gamma:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/output/LayerNorm/gamma:0'%(layer_ndx),
            'bert/transformer_%d/layer_normalization_1/beta:0'%(layer_ndx):'bert/tf_bert_for_pre_training/bert/encoder/layer_._%d/output/LayerNorm/beta:0'%(layer_ndx),
            
        })

    weight_value_tuples = []
    loaded_weights = set()
    skipped_weight_value_tuples = []
    skip_count = 0
    for ndx, (param_value, param) in enumerate(zip(param_values,nezha_params)):
        #param_value为对应的参数值
        #param为对应的参数
        stock_name = transformer_dicts[param.name]
        if stock_name in stock_names:
            ckpt_value = f[stock_name][:]
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
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, resolved_archive_file))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)
    print("Done loading {} BERT weights from: {} into {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), resolved_archive_file, bert,skip_count, len(skipped_weight_value_tuples)))

    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(sorted(stock_names.difference(loaded_weights))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)