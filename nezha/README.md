# Nezha模型结构

Nezha与Bert的区别主要体现在了两个方面：

**1.在Embedding层的时候，Bert的对应结构是word_embeddings+segment_embeddings+position_embeddings，而Nezha的结构为word_embeddings+segment_embeddings**

**2.在AttentionLayer的时候，Bert使用的对应公式与Nezha使用的对应公式不同**

bert使用的对应公式为

![](https://latex.codecogs.com/gif.latex?Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{z}}})V)

nezha使用的对应公式为

![](https://latex.codecogs.com/gif.latex?Attention(Q,K,V) = softmax(\frac{Q(K+\alpha_{ij}^{K})^{T}}{\sqrt{d_{z}}}(V+\alpha_{ij}^{V})))

其中这里面的$\alpha{ij}$实现过程的位置公式为

![](https://latex.codecogs.com/gif.latex?a_{ij}[2k] = sin((j-i)/(10000^{\frac{2k}{d_{z}}})))

![](https://latex.codecogs.com/gif.latex?a_{ij}[2k+1] = cos((j-i)/(10000^{\frac{2k}{d_{z}}})))

对应生成的相对位置编码的代码如下：

```python
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
```

几个需要注意的带有mask的地方(目前未添加上mask的部分)

在NeZhaSelfAttention部分:

```python
if head_mask is not None:
    #这里的head_mask为None,没有被运行
    attention_probs = attention_probs * head_mask
```

在NeZhaEncoder部分：

```python
if self.output_hidden_states:
    #这段在NeZhaLayer之中并没有被运行
    all_hidden_states = all_hidden_states + (hidden_states,)
if self.output_hidden_states:
    #self.output_hidden_states这里的if没有运行
    all_hidden_states = all_hidden_states + (hidden_states,)
```

nezha正式的部分带有decoder部分

```python
if self.config.is_decoder and encoder_hidden_states is not None:
    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None
```

在实验中对nezha模型的思考

1.input_ids不会影响训练过程，即是input_ids只能运行一部分

2.第一次训练完之后全是0，思考是不是学习率过大

3.**易错点:验证集合的内容写错**

```python
res = model.predict([np.array([token_id]),np.array([segment_id])]).argmax(axis=-1)
```

之前错写成为

```
res = model([np.array(token_id),np.array(segment_id)])
```

导致每次res的结果出现了一堆的数据，进而导致结果出错。

原因：外部的大括号是将token_id和segment_id放在一起的，内部的[token_id]中括号表示batch_size = 1，假设放入的长度为32，则[token_id].shape = (1,32)，同理[segment_id].shape = (1,32)，此时才构成一个相应的批次，才能够进行预测。

**4.易错点:测试集合的内容写错**

在中间训练的过程中始终都是token_ids和segment_ids同时放进去

```pytho
res = model.predict([np.array([token_id]),np.array([segment_id])]).argmax(axis=-1)
```

但是在最终预测的时候只有token_ids放入，没有segment_ids放入导致结果出错

```pyth
res = model.predict(np.array([token_id1+token_id2])).argmax(axis=-1)
```

**5.如果某一类一直分数较低的情况下，将这一类预测较好的模型提取出来(可能分数不是最高的)，便于模型融合**

```
Epoch 1/20
2008/2008 [==============================] - 525s 251ms/step - loss: 0.5922 - acc: 0.7593
100%|██████████| 9643/9643 [06:23<00:00, 25.14it/s]
验证集
分类1 准确率
0.9138084236715834
分类1 召回率
0.8785238339313173
分类1 f1_score
0.8958188153310105
分类2 准确率
0.8121761658031088
分类2 召回率
0.786453433678269
分类2 f1_score
0.7991078540704158
分类3 准确率
0.5
分类3 召回率
0.7720465890183028
分类3 f1_score
0.6069326357096141
current final_score = 
0.7738741945792914
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
self.best_score = 
0.7738741945792914
Epoch 2/20
2008/2008 [==============================] - 480s 239ms/step - loss: 0.4182 - acc: 0.8490
100%|██████████| 9643/9643 [05:45<00:00, 27.95it/s]
验证集
分类1 准确率
0.8945062352544658
分类1 召回率
0.9068853579361011
分类1 f1_score
0.9006532620683804
分类2 准确率
0.8429355281207133
分类2 召回率
0.770774537472562
分类2 f1_score
0.8052416052416053
分类3 准确率
0.5699873896595208
分类3 召回率
0.7520798668885191
分类3 f1_score
0.6484935437589671
current final_score = 
0.7879508140445372
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
self.best_score = 
0.7879508140445372
Epoch 3/20
2008/2008 [==============================] - 509s 253ms/step - loss: 0.3350 - acc: 0.8843
100%|██████████| 9643/9643 [06:39<00:00, 24.13it/s]
验证集
分类1 准确率
0.9210665724880805
分类1 召回率
0.891166922945498
分类1 f1_score
0.9058700937825634
分类2 准确率
0.8261548064918851
分类2 召回率
0.8300407651301348
分类2 f1_score
0.8280932269669952
分类3 准确率
0.5695876288659794
分类3 召回率
0.7354409317803661
分类3 f1_score
0.6419753086419754
current final_score = 
0.7943773618992752
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
self.best_score = 
0.7943773618992752
Epoch 4/20
2008/2008 [==============================] - 510s 254ms/step - loss: 0.2606 - acc: 0.9116
100%|██████████| 9643/9643 [06:39<00:00, 24.11it/s]
验证集
分类1 准确率
0.9317896248429366
分类1 召回率
0.8868956090893559
分类1 f1_score
0.9087885154061625
分类2 准确率
0.816
分类2 召回率
0.8316086547507056
分类2 f1_score
0.8237303929181549
分类3 准确率
0.5523114355231143
分类3 召回率
0.7554076539101497
分类3 f1_score
0.6380885453267745
current final_score = 
0.7938467146408171
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 5/20
2008/2008 [==============================] - 510s 254ms/step - loss: 0.1997 - acc: 0.9329
100%|██████████| 9643/9643 [06:44<00:00, 23.86it/s]
验证集
分类1 准确率
0.9257695690413369
分类1 召回率
0.8991969929950453
分类1 f1_score
0.9122898249263304
分类2 准确率
0.8301532686893963
分类2 召回率
0.8322358105989338
分类2 f1_score
0.8311932352020044
分类3 准确率
0.5768725361366623
分类3 召回率
0.7304492512479202
分类3 f1_score
0.6446402349486051
current final_score = 
0.7980889693095815
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
self.best_score = 
0.7980889693095815
Epoch 6/20
2008/2008 [==============================] - 510s 254ms/step - loss: 0.1649 - acc: 0.9443
100%|██████████| 9643/9643 [06:44<00:00, 23.85it/s]
验证集
分类1 准确率
0.9256649638893782
分类1 召回率
0.8978301725610798
分类1 f1_score
0.91153512575889
分类2 准确率
0.8204573547589616
分类2 召回率
0.8325493885230479
分类2 f1_score
0.8264591439688714
分类3 准确率
0.584931506849315
分类3 召回率
0.7104825291181365
分类3 f1_score
0.6416228399699474
current final_score = 
0.7946147805997364
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 7/20
2008/2008 [==============================] - 511s 254ms/step - loss: 0.1320 - acc: 0.9536
100%|██████████| 9643/9643 [06:45<00:00, 23.77it/s]
验证集
分类1 准确率
0.920364848272233
分类1 召回率
0.8964633521271143
分类1 f1_score
0.908256880733945
分类2 准确率
0.8098591549295775
分类2 召回率
0.8294136092819065
分类2 f1_score
0.8195197521301317
分类3 准确率
0.6109467455621301
分类3 召回率
0.6871880199667221
分类3 f1_score
0.6468285043069695
current final_score = 
0.7920934297011922
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 8/20
2008/2008 [==============================] - 512s 255ms/step - loss: 0.1075 - acc: 0.9634
100%|██████████| 9643/9643 [06:45<00:00, 23.81it/s]
验证集
分类1 准确率
0.9212057112638815
分类1 召回率
0.8928754484879549
分类1 f1_score
0.9068193649141072
分类2 准确率
0.8085302239950906
分类2 召回率
0.8262778300407652
分类2 f1_score
0.8173076923076923
分类3 准确率
0.5963431786216596
分类3 召回率
0.7054908485856906
分类3 f1_score
0.6463414634146342
current final_score = 
0.7912435290701639
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 9/20
2008/2008 [==============================] - 514s 256ms/step - loss: 0.0952 - acc: 0.9674
100%|██████████| 9643/9643 [06:47<00:00, 23.68it/s]
验证集
分类1 准确率
0.9225352112676056
分类1 召回率
0.8952673842473945
分类1 f1_score
0.9086967831440215
分类2 准确率
0.7997054491899853
分类2 召回率
0.8513640639698965
分类2 f1_score
0.8247266099635481
分类3 准确率
0.6690140845070423
分类3 召回率
0.632279534109817
分类3 f1_score
0.6501283147989734
current final_score = 
0.7948574927998093
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 10/20
2008/2008 [==============================] - 513s 255ms/step - loss: 0.0775 - acc: 0.9713
100%|██████████| 9643/9643 [06:45<00:00, 23.76it/s]
验证集
分类1 准确率
0.927821997105644
分类1 召回率
0.8763027507261234
分类1 f1_score
0.9013267726913275
分类2 准确率
0.7746558021916269
分类2 召回率
0.8645343367826905
分类2 f1_score
0.8171310017783047
分类3 准确率
0.6654676258992805
分类3 召回率
0.6156405990016639
分类3 f1_score
0.6395851339671564
current final_score = 
0.7869406689048687
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 11/20
2008/2008 [==============================] - 513s 255ms/step - loss: 0.0687 - acc: 0.9750
100%|██████████| 9643/9643 [06:46<00:00, 23.72it/s]
验证集
分类1 准确率
0.9038947900859888
分类1 召回率
0.9159405433111225
分类1 f1_score
0.909877800407332
分类2 准确率
0.8282345442957297
分类2 召回率
0.814989024772656
分类2 f1_score
0.821558400505769
分类3 准确率
0.6777003484320557
分类3 召回率
0.6472545757071547
分类3 f1_score
0.6621276595744681
current final_score = 
0.7979530763435863
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 12/20
2008/2008 [==============================] - 514s 256ms/step - loss: 0.0615 - acc: 0.9771
100%|██████████| 9643/9643 [06:47<00:00, 23.68it/s]
验证集
分类1 准确率
0.9309282088469906
分类1 召回率
0.8773278660515975
分类1 f1_score
0.9033336265282788
分类2 准确率
0.7828182591437483
分类2 召回率
0.865788648479147
分类2 f1_score
0.8222156045265039
分类3 准确率
0.6633333333333333
分类3 召回率
0.6622296173044925
分类3 f1_score
0.6627810158201498
current final_score = 
0.7967506866704713
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 13/20
2008/2008 [==============================] - 513s 256ms/step - loss: 0.0532 - acc: 0.9816
100%|██████████| 9643/9643 [06:47<00:00, 23.65it/s]
验证集
分类1 准确率
0.9191496776441889
分类1 召回率
0.9012472236459935
分类1 f1_score
0.9101104209799862
分类2 准确率
0.8085748792270532
分类2 召回率
0.8397616807776732
分类2 f1_score
0.8238732502691893
分类3 准确率
0.660472972972973
分类3 召回率
0.6505823627287853
分类3 f1_score
0.6554903604358759
current final_score = 
0.7965847587424131
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 14/20
2008/2008 [==============================] - 524s 261ms/step - loss: 0.0526 - acc: 0.9817
100%|██████████| 9643/9643 [06:48<00:00, 23.60it/s]
验证集
分类1 准确率
0.9152981849611063
分类1 召回率
0.9046642747309073
分类1 f1_score
0.9099501632582918
分类2 准确率
0.8108108108108109
分类2 召回率
0.8372530573847601
分类2 f1_score
0.8238198087010181
分类3 准确率
0.6530973451327433
分类3 召回率
0.6139767054908486
分类3 f1_score
0.6329331046312178
current final_score = 
0.7890892727890781
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 15/20
2008/2008 [==============================] - 524s 261ms/step - loss: 0.0468 - acc: 0.9835
100%|██████████| 9643/9643 [06:47<00:00, 23.68it/s]
验证集
分类1 准确率
0.9108454312553373
分类1 召回率
0.9111566717922432
分类1 f1_score
0.9110010249402117
分类2 准确率
0.82356608478803
分类2 召回率
0.8284728755095642
分类2 f1_score
0.8260121932155698
分类3 准确率
0.6637931034482759
分类3 召回率
0.6405990016638935
分类3 f1_score
0.6519898391193903
current final_score = 
0.7963818028591684
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 16/20
2008/2008 [==============================] - 524s 261ms/step - loss: 0.0455 - acc: 0.9845
100%|██████████| 9643/9643 [06:47<00:00, 23.65it/s]
验证集
分类1 准确率
0.9175168597613695
分类1 召回率
0.9065436528276097
分类1 f1_score
0.9119972499140597
分类2 准确率
0.8263867126292698
分类2 召回率
0.8269049858889934
分类2 f1_score
0.8266457680250784
分类3 准确率
0.6203288490284006
分类3 召回率
0.6905158069883528
分类3 f1_score
0.6535433070866142
current final_score = 
0.7978203546833055
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 17/20
2008/2008 [==============================] - 523s 260ms/step - loss: 0.0397 - acc: 0.9860
100%|██████████| 9643/9643 [06:48<00:00, 23.62it/s]
验证集
分类1 准确率
0.91431520991053
分类1 召回率
0.9079104732615753
分类1 f1_score
0.9111015859408488
分类2 准确率
0.8222292390274708
分类2 召回率
0.8165569143932268
分类2 f1_score
0.8193832599118943
分类3 准确率
0.5858433734939759
分类3 召回率
0.6472545757071547
分类3 f1_score
0.6150197628458498
current final_score = 
0.7821793771658362
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 18/20
2008/2008 [==============================] - 524s 261ms/step - loss: 0.0418 - acc: 0.9855
100%|██████████| 9643/9643 [06:46<00:00, 23.70it/s]
验证集
分类1 准确率
0.9058288409703504
分类1 召回率
0.9186741841790534
分类1 f1_score
0.9122062940028841
分类2 准确率
0.8267419962335216
分类2 召回率
0.825964252116651
分类2 f1_score
0.8263529411764706
分类3 准确率
0.6775431861804223
分类3 召回率
0.5873544093178037
分类3 f1_score
0.6292335115864528
current final_score = 
0.7899888461959567
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 19/20
2008/2008 [==============================] - 526s 262ms/step - loss: 0.0391 - acc: 0.9871
100%|██████████| 9643/9643 [06:48<00:00, 23.61it/s]
验证集
分类1 准确率
0.9010135135135136
分类1 召回率
0.911327524346489
分类1 f1_score
0.9061411704748152
分类2 准确率
0.8350819672131148
分类2 召回率
0.7986829727187206
分类2 f1_score
0.8164769995191538
分类3 准确率
0.6092124814264487
分类3 召回率
0.6821963394342762
分类3 f1_score
0.6436420722135008
current final_score = 
0.7893083378733369
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
Epoch 20/20
2008/2008 [==============================] - 521s 259ms/step - loss: 0.0361 - acc: 0.9873
100%|██████████| 9643/9643 [06:48<00:00, 23.58it/s]
验证集
分类1 准确率
0.907749703238935
分类1 召回率
0.914573722877157
分类1 f1_score
0.9111489361702126
分类2 准确率
0.8228321896444167
分类2 召回率
0.8272185638131075
分类2 f1_score
0.8250195465207194
分类3 准确率
0.6851851851851852
分类3 召回率
0.6156405990016639
分类3 f1_score
0.6485539000876425
current final_score = 
0.7953247051710045
evaluate sparse_categorical_accuracy = 
tf.Tensor(0.60696876, shape=(), dtype=float32)
#######################################
```

**6.思考：为什么类别少的不能一味地增大该类别的数据？**

增大类别之后可以增大对应的召回率(对于一组数据，更偏向于预测它是这个类别稀少的类别的数据)

但是有可能减少准确率(在你预测的所有这个类别的数据之中，预测对的数据太少了)

因此，对于类别较为少的数据，不能一味地增大数据，否则有可能数据增大之后效果并不一定好，可以尝试适度地增加数据。

如果准确率远远低于召回率，减少数据，如果准确率远远高于召回率，增加数据。

(对于预测不好的类别，可以尝试减少数据，不变数据，增加数据三种相应的操作方式)

但是仔细观察数据，发现准确率和召回率一直在动态变化的过程之中，并不是准确率一直大于召回率或者准确率一直小于召回率的过程