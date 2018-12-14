import numpy as np
import tensorflow as tf

LSTM_KEEP_PROB = 0.9  # LSTM不被dropout的概率
HIDDEN_SIZE = 200  # 隐藏层大小
NUM_LAYERS = 2
VOCAB_SIZE = 10000
EMBEDDING_KEEP_PROB = 0.9  # 词向量不被dropout的概率

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
        self.output_data = tf.placeholder(tf.int32,[batch_size,num_steps])

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


def main():
    read_data("test.txt")

if __name__ == "__main__":
    main()

