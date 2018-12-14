with open("ptb.vocab.txt",'r',encoding = 'utf-8') as text:
    list_vocab = [w.strip() for w in text]
dict_vocab = {k:v for (k,v) in zip(list_vocab,range(len(list_vocab)))}
# print(dict_vocab)
def get_id(word):  # 得到单词的id
    return dict_vocab[word] if word in list_vocab else dict_vocab["<unk>"]

fin = open("ptb.train.txt",'r',encoding='utf-8')  # 原始单词文件
fout = open('ptb.vocab','w',encoding = 'utf-8')  # 变为单词编号的文件
words = []
for i in fin:
    words +=i.strip().split() + ['<eos>']
    outline = " ".join([str(get_id(j)) for j in words])
    fout.write(outline)
fin.close()
fout.close()


