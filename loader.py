from models import Bert
import tensorflow as tf
import tensorflow.keras as keras
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
    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    #之前使用param相当于重新定义了一个属性param属性，然后通过param属性去查找相应的
    #适配其他类型的权重实际上就是修改对应的参数名称，让它能够跟新的权重内容匹配上
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
            
