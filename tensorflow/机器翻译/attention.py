import tensorflow as tf

self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

with tf.variable_scope("encoder"):
    enc_outputs,enc_state = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,self.enc_cell_bwm,src_emb,src_size,dtype=tf.float32
    )
    # 将两个LSTM输出拼接为一个张量
    enc_outputs = tf.concat(enc_outputs[0],enc_outputs[1])

with tf.variable_scope("decoder"):
    # 选择注意力机制的权重计算模型
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        HIDDEN_SIZE,enc_outputs,memory_sequence_length = src_size
    )
    # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        self.dec_cell,attention_mechanism,attention_layer_size = HIDDEN_SIZE
    )
    # 使用attention_cell和dynamic_rnn构造编码器，
    # 注意这里没有指定init_state,也就是没有使用编码器的输出来初始化输入，而完全依赖注意力作为信息来源
    dec_outputs,_=tf.nn.dynamic_rnn(attention_cell,trg_emb,trg_size,dtype=tf.float32)


