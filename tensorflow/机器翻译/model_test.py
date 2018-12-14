import tensorflow as tf
# 读取checkpoint路径
CHECKPOINT_PATH = 'seq2seq_ckpt-9000'

# 模型参数，与训练时模型参数保持一致
HIDDEN_SIZE = 1024

# 词汇表中<sos>和<eos>的ID,在解码过程中需要用<sos>作为第一步的输入，并将检查是否是<eos>
SOS_ID = 1
EOS_ID = 2


# 定义类描述模型
class NMTModel(object):
    def __init__(self):
        pass  # 与训练时的init函数相同，在训练和解码程序中复用NMTModel类及init函数，这样变量定义相同。

    def inference(self,src_input):
        src_size = tf.convert_to_tensor([len(src_input)],dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input],dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)

        # 使用dynamic_rnn构造编码器
        with tf.variable_scope("encode"):
            enc_outputs,enc_state = tf.nn.dynamic_rnn(
                self.enc_cell,src_emb,src_size,dtype = tf.float32
            )

        # 设置解码的最大步骤（避免无限循环）
        MAX_DEC_LEN = 100

        with tf.variable_scope('decoder/rnn/multi_rnn_cell'):
            init_array = tf.TensorArray(dtype=tf.int32,size=0,
                                        dynamic_size=True, clear_after_read=False)  # 用于存储生成的句子，可动态变长
            init_array = init_array.write(0,SOS_ID)  # 填入第一个单词<sos>作为解码器的输入
            init_loop_var = (enc_state,init_array,0)  # 循环的初始状态，隐藏状态，生成的句子，解码步数
            def continue_loop_condition(state,trg_ids,step):
                return tf.reduce_all(
                    tf.logical_and(tf.not_equal(trg_ids.read(step),EOS_ID),tf.less(step,MAX_DEC_LEN))
                )
            def loop_body(state,trg_ids,step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,trg_input)
                dec_outputs,next_state = self.dec_cell.call(state = state,input = trg_emb)  # 调用dec_cell向前计算一步
                outputs = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
                logits = (tf.matmul(outputs,self.softmax_weight)+self.softmax_bias)
                next_id = tf.arg_max(logits,axis=1,output_type = tf.int32)
                trg_ids = trg_ids.write(step+1,next_id[0])
                return next_state,trg_ids,step+1
            state,trg_ids,step = tf.while_loop(continue_loop_condition,loop_body,init_loop_var)
            return trg_ids.stack()

def main():
    with tf.variable_scope('nmt_model',reuse = None):
        model = NMTModel()
    test_sentence = [90,13,9,689,4,2]  # 测试的例子
    output_op = model.inference(test_sentence)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,CHECKPOINT_PATH)
    # 读取翻译结果
    output = sess.run(output_op)
    print(output)
    sess.close()

if __name__ == "__main__":
    main()














