vocab_output = 'ptb.vocab.txt'
word = {}
with open("ptb.train.txt",'r',encoding="utf-8") as text:
    for line in text:
        # print(line.strip().split())
        for vocab in line.strip().split():
            if vocab in word:   # 一定要用这个if判断一下，不能直接用word[vocab] +=1，否则会报错，因为原先的word里面没有vocab
                word[vocab] +=1
            else:
                word[vocab] =1
word_sorted = sorted(word.items(),key = lambda i:i[1],reverse=True)  # 排序，其实不排也可以，得到刚才的词频字典即可
print(word_sorted)
all_words = [i[0] for i in word_sorted]  # 得到所有不同的词
print(all_words)
all_words = ['<eos>']+all_words
# all_words = ['<unk>','<sos>','<eos>']+all_words   # 如有需要，加几个需要的词
# 如有需要，这里可以再加一下筛选低频词的的操作，比如只选出前1000个高频词作为词典的词
with open('ptb.vocab.txt','w',encoding='utf-8') as f:
    for i in all_words:
        f.write(i+'\n')  # 此时，文件中词的行数就是词的编号


