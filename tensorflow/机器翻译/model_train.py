import tensorflow as tf

MAX_LEN = 50
SOS_ID = 1

def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda x: tf.string_split([x]).values)  # 按空格分开，变成一个list
    dataset = dataset.map(lambda x: tf.string_to_number(x,tf.int32))  # 每个字符串转换为int数字
    dataset = dataset.map(lambda x: (x,tf.size(x)))  # 每一行数据即包含了数字，也包含了这一行数字的个数，组成一个tuple
    return dataset

def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    dataset = tf.data.Dataset.zip((src_data,trg_data))

    def FilterLength(src_tuple,trg_tuple):
        ((src_input,src_len),(trg_label,trg_len)) = (src_tuple,trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len,1),tf.less_equal(src_len,MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len,1),tf.less_equal(trg_len,MAX_LEN))
        return tf.logical_and(src_len_ok,trg_len_ok)
    dataset = dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple,trg_tuple):
        ((src_input,src_len),(trg_label,trg_len)) = (src_tuple,trg_tuple)
        trg_input = tf.concat([[SOS_ID],trg_label[:-1]],axis = 0)
        return ((src_input,src_len),(trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    dataset = dataset.shuffle(10000)
    padded_shapes = (
        (tf.TensorShape([None]),
         tf.TensorShape([])),
        (tf.TensorShape([None]),
         tf.TensorShape([None]),
         tf.TensorShape([]))
    )
    batch_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batch_dataset

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOB_SIZE = 10000
TRG_VOB_SIZE = 4000
SHARE_EMB_AND_SOFTMAX = 1
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SRC_TRAIN_DATA='train.en'
TRG_TRAIN_DATA='train.zh'
BATCH_SIZE = 100
NUM_EPOCH = 5
CHECKPOINT_PATH = "seq2seq_ckpt"

# 定义一个类来描述模型
class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)]
        )
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)]
        )

        # 定义词向量
        self.src_embedding = tf.get_variable(
            'src_emb',[SRC_VOB_SIZE,HIDDEN_SIZE]
        )
        self.trg_embedding = tf.get_variable(
            'trg_emb',[TRG_VOB_SIZE,HIDDEN_SIZE]
        )

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight",[HIDDEN_SIZE,TRG_VOB_SIZE]
            )
        self.softmax_bias = tf.get_variable(
            "softmax_bias",[TRG_VOB_SIZE]
        )

        # 定义前向计算图，forward函数
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
        # 将输入和输出单词编号转换为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding,trg_input)
        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb,KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb,KEEP_PROB)
        # 使用dynamic_rnn构造编码器
        with tf.variable_scope("encoder"):
            enc_outputs,enc_state = tf.nn.dynamic_rnn(
                self.enc_cell,src_emb,src_size,dtype = tf.float32
            )
        # 使用dynamic_rnn构造解码器
        with tf.variable_scope("decoder"):
            dec_outputs,_ = tf.nn.dynamic_rnn(
                self.dec_cell,trg_emb,trg_size,initial_state=enc_state
            )
        # 计算解码器每一步的log perplexity，与语言模型相同
        output = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
        logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = tf.shape(trg_label,[-1]), logits = logits
        )
        # 计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练
        label_weights = tf.sequence_mask(trg_size,maxlen = tf.shape(trg_label)[1],dtype=tf.float32)
        label_weights = tf.reshape(label_weights,[-1])
        cost = tf.reduce_sum(loss*label_weights)
        cost_per_token = cost/tf.reduce_sum(label_weights)

        # 定义反向传播
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(cost/tf.to_float(batch_size),trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads,MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.apply_gradients(zip(grads,trainable_variables))
        return cost_per_token,train_op

def run_epoch(session,cost_op,train_op,saver,step):
    while True:
        try:
            cost, _ = session.run([cost_op,train_op])
            if step%10 == 0:
                print("after %d steps, per token cost is %.3f" % (step,cost))
            if step%200 == 0:
                saver.save(session,CHECKPOINT_PATH,global_step= step)
            step +=1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    # 定义训练用的循环神经网络
    with tf.variable_scope("nmt_model",reuse=None,initializer = initializer):
        train_model = NMTModel()
    # 定义输入数据
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA,TRG_TRAIN_DATA,BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()
    # 定义前向计算图，输入数据以张量形式提供给forwoard函数
    cost_op,train_op = train_model.forward(src,src_size,trg_input,trg_label,trg_size)
    # 训练模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i+1))
            sess.run(iterator.initializer)
            step = run_epoch(sess,cost_op,train_op,saver,step)

if __name__ == "__main__":
    main()









