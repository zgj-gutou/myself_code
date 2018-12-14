import numpy as np
import tensorflow as tf

LSTM_KEEP_PROB = 0.9  # LSTM不被dropout的概率
HIDDEN_SIZE = 200  # 隐藏层大小
NUM_LAYERS = 2
VOCAB_SIZE = 10000
EMBEDDING_KEEP_PROB = 0.9  # 词向量不被dropout的概率
SHARE_EMB_AND_SOFTMAX = True
MAX_GRAD_NORM = 5
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
TRAIN_DATA = "ptb.train"
EVAL_DATA = "ptb.valid"
TEST_DATA = "ptb.test"
NUM_EPOCH = 5

# 整理成batch
def read_data(path_file):
    str_all_word_id = ''
    with open(path_file,'r',encoding = 'utf-8') as text:
        for i in text.readlines():
            str_all_word_id += " "+(i.strip())  # 变成一个长字符串
    list_all_word_id = [int(i) for i in str_all_word_id.split()]  # 转为int型数字
    return list_all_word_id

def make_batches(list_all_word_id,batch_size,num_step):
    num_batches = (len(list_all_word_id)-1)//(batch_size*num_step)

    data = list_all_word_id[:num_batches*batch_size*num_step] # 训练集数据
    data = np.reshape(data,[batch_size,num_batches*num_step])
    data_batches = np.split(data,num_batches,axis = 1)

    label = list_all_word_id[1:num_batches*batch_size*num_step+1]  # 标签
    label = np.reshape(label,[batch_size,num_batches*num_step])
    label_batches = np.split(label,num_batches,axis = 1)

    return list(zip(data_batches,label_batches))  # 元素为一个元祖的list

# 定义模型
class PTBModel():
    def __init__(self,is_training,batch_size,num_steps):
        # 定义batch大小（每一batch的单词个数）和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义预设的输入和输出，维度都是[batch_size,num_steps]
        self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells =[tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
            output_keep_prob=dropout_keep_prob
        ) for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)  # 深层神经网络

        # 初始化最初的状态，即全零的向量，这个量只在每个epoch初始化第一个batch时使用
        self.initial_state = cell.zero_state(batch_size,tf.float32)

        # 定义单词的词向量矩阵
        embedding = tf.get_variable("embedding",[VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转换为词向量
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)

        # 只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs,EMBEDDING_KEEP_PROB)
            outputs = []
            state = self.initial_state
            with tf.variable_scope("RNN"):
                for time_step in range(num_steps):  # num_steps表示截断个数，也表示输入的第二个维度
                    if time_step>0:
                        tf.get_variable_scope().reuse_variables()
                    cell_putput, state = cell(input[:,time_step,:],state)
                    outputs.append(cell_putput)
            output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

            # Softmax层：将RNN在每个位置上的输出转化为各个单词的logits。
            if SHARE_EMB_AND_SOFTMAX:
                weight = tf.transpose(embedding)
            else:
                weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
            bias = tf.get_variable("bias", [VOCAB_SIZE])
            logits = tf.matmul(output, weight) + bias
            #         print("logits:",logits)

            # 定义交叉熵损失函数和平均损失。
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]),
                logits=logits)
            print("loss:", loss)
            self.cost = tf.reduce_sum(loss) / batch_size
            self.final_state = state

            # 只在训练模型时定义反向传播操作。
            if not is_training: return

            trainable_variables = tf.trainable_variables()
            print("trainable_variables:", trainable_variables)
            # 控制梯度大小，定义优化方法和训练步骤。
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables))

# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值。
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量。
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 训练一个epoch。
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单
        # 词为给定单词的概率。
        cost, state, _ = session.run(
             [model.cost, model.final_state, train_op],
             {model.input_data: x, model.targets: y,
              model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        # 只有在训练时输出日志。
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                  step, np.exp(total_costs / iters)))
        step += 1

    # 返回给定模型在给定数据上的perplexity值。
    return step, np.exp(total_costs / iters)


# 从文件中读取数据，并返回包含单词编号的数组。
def read_data(file_path):
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串。
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    return id_list


def make_batches(id_list, batch_size, num_step):
    # 计算总的batch数量。每个batch包含的单词数量是batch_size * num_step。
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # 如9-4图所示，将数据整理成一个维度为[batch_size, num_batches * num_step]
    # 的二维数组。
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成num_batches个batch，存入一个数组。
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位。这里得到的是RNN每一步输出所需要预测的
    # 下一个单词。
    label = np.array(id_list[1 : num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)
    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵。
    return list(zip(data_batches, label_batches))


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model",
                           reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试用的循环神经网络模型。它与train_model共用参数，但是没有dropout。
    with tf.variable_scope("language_model",
                           reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(
            read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        eval_batches = make_batches(
            read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        test_batches = make_batches(
            read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)

        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session, train_model, train_batches,
                                         train_model.train_op, True, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches,
                                     tf.no_op(), False, 0)
            print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))

        _, test_pplx = run_epoch(session, eval_model, test_batches,
                                 tf.no_op(), False, 0)
        print("Test Perplexity: %.3f" % test_pplx)


if __name__ == "__main__":
    main()


