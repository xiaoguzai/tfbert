# Nezha模型结构

Nezha与Bert的区别主要体现在了两个方面：

**1.在Embedding层的时候，Bert的对应结构是word_embeddings+segment_embeddings+position_embeddings，而Nezha的结构为word_embeddings+segment_embeddings**

**2.在AttentionLayer的时候，Bert使用的对应公式与Nezha使用的对应公式不同**

bert使用的对应公式为

$Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{z}}})V$

nezha使用的对应公式为

$Attention(Q,K,V) = softmax(\frac{Q(K+\alpha_{ij}^{K})^{T}}{\sqrt{d_{z}}})(V+\alpha_{ij}^{V})$

其中这里面的$\alpha{ij}$实现过程的位置公式为

$a_{ij}[2k] = sin((j-i)/(10000^{\frac{2k}{d_{z}}}))$

$a_{ij}[2k+1] = cos((j-i)/(10000^{\frac{2k}{d_{z}}}))$

对应生成的相对位置编码的代码如下：

```python
def _generate_relative_positions_matrix(length, max_relative_position, cache=False):
    if not cache:
        range_vec = tf.range(length)
        range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
        distance_mat_clipped = range_mat - tf.transpose(range_mat)
        distance_mat_clipped = tf.clip_by_value(distance_mat_clipped,-max_relative_position,
                                                max_relative_position)
    else:
        distance_mat = tf.expand_dims(tf.range(-length+1, 1, 1), 0)
        distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def relative_positions_encodings(length, depth, max_relative_position, cache=False):
    relative_positions_matrix = _generate_relative_positions_matrix(
        length, max_relative_position, cache=cache)
    vocab_size = max_relative_position * 2 + 1
    embeddings_table = np.zeros([vocab_size, depth]) #range(vocab_size * depth)#tf.get_variable(name="embeddings", shape=[vocab_size, depth], initializer=create_initializer())
    position = tf.range(0.0, vocab_size, 1.0)#.unsqueeze(1)
    position = tf.reshape(position, [vocab_size, -1])

    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = tf.convert_to_tensor(embeddings_table, tf.float32)
    flat_relative_positions_matrix = tf.reshape(relative_positions_matrix, [-1])
    one_hot_relative_positions_matrix = tf.one_hot(flat_relative_positions_matrix, depth=vocab_size)
    embeddings = tf.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
    my_shape = relative_positions_matrix.shape.as_list()
    my_shape.append(depth)

    embeddings = tf.reshape(embeddings, my_shape)
    return embeddings
```

