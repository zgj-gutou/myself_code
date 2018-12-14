from pyhanlp import *
import pandas as pd
import jieba.posseg as pseg

# 选手需为每一篇文章预测相应的关键词，选手提交的预测结果中，每篇文章最多输出两个关键词。
# 预测结果跟标注结果命中一个得 0.5 分，命中两个得一分。英文关键词不区分大小写。

# 思路：1、用多种取关键词的方法取关键词，再取各种方法所得到的关键词的交集
# 2、去掉标点符号等

# data0 = pd.read_csv('../result/jieb_ruler_result_1011_1.csv')
# data0.to_csv('../result/jieb_ruler_result_1011_1_1.csv',encoding='utf_8_sig',index=None)

files = open('C:/pycharm_project/神策杯数据集/stopwords_zgj.txt', "r",encoding='utf-8')  # 读取文件，停用词
stop_words = files.readlines()
stopws = []
for line in stop_words:
    sw = line.strip('\n')
    stopws.append(sw)
files.close()
print(stopws)
print(len(stopws))

files = open('C:/pycharm_project/神策杯数据集/stars.txt', "r",encoding='utf-8')  # 读取文件，明星人名
stars_words = files.readlines()
stars = []
for line in stars_words:
    sw1 = line.strip('\n')
    stars.append(sw1)
files.close()
print(stars)
print(len(stars))

# ID  标题 文本内容
data = pd.read_csv('../all_docs.txt',sep='\001',header=None)  # 108295篇资讯文章数据，数据格式为：ID 文章标题 文章正文，中间由\001分割。
data.columns = ['id','title','doc']

train = pd.read_csv('../train_docs_keywords.txt',sep='\t',header=None)  # 1000篇文章的关键词标注结果，数据格式为：ID 关键词列表，中间由\t分割。
train.columns = ['id','label']

a = list(train['label'])
print(a)
CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
for i in range(len(a)):
    b = a[i].split(',')
    print(b)
    for j in b:
        CustomDictionary.add(j)  # 把训练集的关键词添加到hanlp的词典中

for i in range(len(stars)):  # 添加明星
    b = stars[i].split(',')
    print(b)
    for j in b:
        CustomDictionary.add(j)  # 把明星人名添加到hanlp的词典中

print(data)
print("------------------")
print(train)

train_id_list = list(train['id'].unique())
train_title_doc = data[data['id'].isin(train_id_list)]  # 训练集
test_title_doc = data[~data['id'].isin(train_id_list)]  # 测试集
train_title_doc = pd.merge(train_title_doc,train,on=['id'],how='inner')  # 合并，从左到右依次为id,label,title,doc

print("------test_title_doc------")
print(test_title_doc['doc'])

import jieba
import re
import jieba.analyse
import numpy as np

# 去除文章的数字，数字没有意义，单纯一个数字，不能达到对文章内容的区分,这里只用了title题目，没用doc文章
train_title_doc['title_cut'] = train_title_doc['title'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890', x)))
# 去除文章的数字，数字没有意义，单纯一个数字，不能达到对文章内容的区分,这里用了doc文章
train_title_doc['doc_cut'] = train_title_doc['doc'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890', x)))

# 策略 extract_tags 直接利用jieba的提取主题词的工具
train_title_doc['title_cut_1'] = train_title_doc['title_cut'].apply(lambda x:','.join(jieba.analyse.extract_tags(x,topK = 5,allowPOS=("ns", "n", "vn", "v", "nr"))))
# 策略 textrank 直接利用jieba的提取主题词的工具
train_title_doc['doc_cut'] = train_title_doc['doc_cut'].apply(lambda x:','.join(jieba.analyse.textrank(x,topK = 5,allowPOS=("ns", "n", "vn", "v", "nr"))))

# 根据句法依存，如主语，宾语，但后面没用到
train_title_doc['title_cut_2'] = train_title_doc['title_cut'].apply(lambda x:HanLP.parseDependency(x))

# 分割标题，同时得到各词的词性,如人名，地名，专有名词
train_title_doc['title_cut_3'] = train_title_doc['title_cut'].apply(lambda x:HanLP.segment(x))

# THULAC分割标题各词的词性,如人名，地名，专有名词，后面没用到
import thulac
thu1 = thulac.thulac()  #默认模式
train_title_doc['title_cut_4'] = train_title_doc['title_cut'].apply(lambda x:thu1.cut(x))

# 第二规则 提取 《》 通过分析发现，凡是书名号的东西都会被用来作为主题词
train_title_doc['title_regex'] = train_title_doc['title'].apply(lambda x:','.join(re.findall(r"《(.+?)》",x)))

# 利用策略 + 规则 查看训练集的准确率
train_offline_result = train_title_doc[['id','label','title_cut_1','title_cut_2','title_cut_3','doc_cut','title_regex','title_cut_4']]

label1 = []
label2 = []
# 验证我这个规则能够达到的分数 记得 * 0.5
count = 0
for i in train_offline_result.values:
    flag = 0
    flag_name = 0
    flag_pro = 0
    tmp_result = ''
    result = str(i[1]).split(',')   # 训练集的label列
    title_cut_1 = str(i[2]).split(',')   # title_cut_1列
    title_cut_2 = i[3]  # title_cut_2列
    title_cut_3 = i[4]  # title_cut_3列
    doc_cut = str(i[5]).split(',')  # doc_cut列
    title_regex = str(i[6]).split(',')  # title_regex列
    title_cut_4 = i[7]

    print('title_cut_1:', title_cut_1)
    print('doc_cut:', doc_cut)

    # 所有规则整合到一起
    if len(title_regex) >= 2:  # 如果有两个《》
        tmp_result = title_regex
        count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
        flag = 1
    elif len(title_regex) == 1 and title_regex[0] != '':  # 如果只有一个《》
        print("title_regex:", title_regex)
        tmp_result = title_regex[0]
        print("title_regex[0]:", title_regex[0])
        for words_title_cut_3 in title_cut_3:  # 遍历看是否有人名
            if flag == 0:
                if str(words_title_cut_3.nature) in ['nr', 'nrf', 'nrj', 'nr2','nz']:  # 先判断真正的人名
                    if len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result and words_title_cut_3.word not in tmp_result \
                            and tmp_result not in words_title_cut_3.word and words_title_cut_3.word not in stopws:
                        tmp_result = [tmp_result, words_title_cut_3.word]  # 由《》和人名组成
                        print(tmp_result)
                        count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                        flag = 1
                        break  # 用于跳出for循环
        if flag == 0:  # 如果找不到人名
            for words_title_cut_3 in title_cut_3:  # 遍历寻找专业名词（地名等）
                if flag == 0:
                    if str(words_title_cut_3.nature) in ['ni', 'nic', 'nis', 'nit', 'ns', 'nsf', 'nt', 'ntc', 'ntcb',
                                                         'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
                                                         'nl','nm','nmc','nn','nb','nba','nbc','nbp','nf',
                                                         'ng','nh','nhd','nhm']:
                        if len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                                and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word\
                                and words_title_cut_3.word not in stopws:
                            tmp_result = [tmp_result, words_title_cut_3.word]  # 由《》和专业名词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                            break  # 用于跳出for循环
        if flag == 0:  # 如果找不到人名和其他专业名词，则把提取的题目中的第一个关键字拿来（注意，这个关键字不能和《》内的重复
            if (len(title_cut_1) > 0 and tmp_result != title_cut_1[0] and tmp_result not in title_cut_1[0] and
                    title_cut_1[0] not in tmp_result and title_cut_1[0] not in stopws):
                tmp_result = [tmp_result, title_cut_1[0]]   # 由《》和题目关键字组成
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
            elif (len(title_cut_1) > 1 and tmp_result != title_cut_1[1] and tmp_result not in title_cut_1[1] and
                  title_cut_1[1] not in tmp_result and title_cut_1[1] not in stopws):
                tmp_result = [tmp_result, title_cut_1[1]]   # 由《》和题目关键字组成
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
            elif (len(title_cut_1) > 2 and tmp_result != title_cut_1[2] and tmp_result not in title_cut_1[2] and
                  title_cut_1[2] not in tmp_result and title_cut_1[2] not in stopws):
                tmp_result = [tmp_result, title_cut_1[2]]  # 由《》和题目关键字组成
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
            elif (len(doc_cut) > 0 and tmp_result != doc_cut[0] and tmp_result not in doc_cut[0] and
                  doc_cut[0] not in tmp_result and doc_cut[0] not in stopws):
                tmp_result = [tmp_result, doc_cut[0]]    # 由《》和文章关键字组成
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
            elif (len(doc_cut) > 1 and tmp_result != doc_cut[1] and tmp_result not in doc_cut[1] and
                  doc_cut[1] not in tmp_result and doc_cut[1] not in stopws):
                tmp_result = [tmp_result, doc_cut[1]]   # 由《》和文章关键字组成
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
            elif (len(doc_cut) > 2 and tmp_result != doc_cut[2] and tmp_result not in doc_cut[2] and
                  doc_cut[2] not in tmp_result and doc_cut[2] not in stopws):
                tmp_result = [tmp_result, doc_cut[2]]   # 由《》和文章关键字组成
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
            else:
                tmp_result = [tmp_result, "朱国杰"]   # 由《》和随便取的名字组成，因为可能存在很特殊的情况，所以加了这个
                print(tmp_result)
                count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                flag = 1
    elif len(title_regex) == 1 and flag == 0 and title_regex[0] == '':  # 没有《》
        tmp_result = ""  # 其实上一个if应该是能保证有两个关键词的，以防万一，这里又重新初始化
        for words_title_cut_3 in title_cut_3:
            if flag == 0:
                if str(words_title_cut_3.nature) in ['nr', 'nrf', 'nrj', 'nr2','nz']:
                    if tmp_result == '' and len(words_title_cut_3.word) > 1 and words_title_cut_3.word not in stopws:  # 找到一个人名
                        tmp_result = words_title_cut_3.word
                        print('words_title_cut_3.word:',words_title_cut_3.word)
                        flag_name = 1  # 标志着找到一个人名
                    elif tmp_result != '' and len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                            and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word \
                            and words_title_cut_3.word not in stopws:
                        tmp_result = [tmp_result, words_title_cut_3.word]   # 由两个人名组成
                        print(tmp_result)
                        count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                        flag = 1
                        flag_name = 2  # 标志着找到两个人名
                        break  # 用于跳出for循环
        if flag_name == 1 and flag == 0 and tmp_result != '':  # 能到这个if,说明遍历过了，但是只找到一个人名
            for words_title_cut_3 in title_cut_3:  # 在出现了一个人名的情况下，重新遍历,看是否有专业名词
                if flag == 0:
                    if str(words_title_cut_3.nature) in ['ni', 'nic', 'nis', 'nit', 'ns', 'nsf', 'nt', 'ntc', 'ntcb',
                                                         'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
                                                         'nl','nm','nmc','nn','nb','nba','nbc','nbp','nf',
                                                         'ng','nh','nhd','nhm']:
                        if len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                                and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word \
                                and words_title_cut_3.word not in stopws:
                            tmp_result = [tmp_result, words_title_cut_3.word]  # 由一个人名和一个专业名词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                            break  # 用于跳出for循环
            if flag == 0:  # 说明没有专业名词，则取题目关键词中的第一个
                if (len(title_cut_1) > 0 and tmp_result != title_cut_1[0] and tmp_result not in title_cut_1[0] and
                        title_cut_1[0] not in tmp_result and title_cut_1[0] not in stopws):
                    tmp_result = [tmp_result, title_cut_1[0]]   # 由一个人名和题目关键字组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) > 1 and tmp_result != title_cut_1[1] and tmp_result not in title_cut_1[1] and
                      title_cut_1[1] not in tmp_result and title_cut_1[1] not in stopws):
                    tmp_result = [tmp_result, title_cut_1[1]]  # 由一个人名和题目关键字组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) == 1 and title_cut_1[0] == tmp_result and len(doc_cut)>0 and len(doc_cut[0])>1
                      and tmp_result != doc_cut[0] and tmp_result not in doc_cut[0] and
                      doc_cut[0] not in tmp_result and doc_cut[0] not in stopws):
                    tmp_result = [tmp_result, doc_cut[0]]   # 由一个人名和文章关键字组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) == 1 and title_cut_1[0] == tmp_result and len(doc_cut)>1 and
                      tmp_result != doc_cut[1] and tmp_result not in doc_cut[1] and
                      doc_cut[1] not in tmp_result and doc_cut[1] not in stopws):
                    tmp_result = [tmp_result, doc_cut[1]]   # 由一个人名和文章关键字组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                else:  # 出现一些特殊情况，比如人名或者专业名词很长，涵盖了几个比较重要的词，或者正文doc很短很短，这里想改还可以改，不过情况比较多！！！
                    tmp_result = [tmp_result, doc_cut[0]]   # 这里的doc_cut[0]有可能是''，# 由一个人名和文章关键字组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
        if flag_name == 0 and flag == 0:  # 说明没有人名，则重新挑专业名词
            tmp_result = ""
            for words_title_cut_3 in title_cut_3:
                if flag == 0:
                    if str(words_title_cut_3.nature) in ['ni', 'nic', 'nis', 'nit', 'ns', 'nsf', 'nt', 'ntc', 'ntcb',
                                                         'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
                                                         'nl','nm','nmc','nn','nb','nba','nbc','nbp','nf',
                                                         'ng','nh','nhd','nhm']:
                        if tmp_result == '' and len(words_title_cut_3.word) > 1 and words_title_cut_3.word not in stopws:
                            tmp_result = words_title_cut_3.word
                            flag_pro = 1  # 说明找到一个专业名词
                        elif tmp_result != '' and len(
                                words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                                and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word \
                                and words_title_cut_3.word not in stopws:
                            tmp_result = [tmp_result, words_title_cut_3.word]  # 由两个专业名词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                            flag_pro = 2  # 说明找到两个专业名词
                            break  # 用于跳出for循环
            if flag_pro == 1 and flag == 0 and tmp_result != '':  # 说明只找到一个专业名词，则再取关键词中的一个
                if (len(title_cut_1) > 0 and tmp_result != title_cut_1[0] and tmp_result not in title_cut_1[0] and
                        title_cut_1[0] not in tmp_result and title_cut_1[0] not in stopws and title_cut_1[0]!=''):
                    tmp_result = [tmp_result, title_cut_1[0]]    # 由一个专业名词和一个题目关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) > 1 and tmp_result != title_cut_1[1] and tmp_result not in title_cut_1[1] and
                      title_cut_1[1] not in tmp_result and title_cut_1[1] not in stopws and title_cut_1[1]!=''):
                    tmp_result = [tmp_result, title_cut_1[1]]  # 由一个专业名词和一个题目关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(doc_cut) > 0 and tmp_result != doc_cut[0] and tmp_result not in doc_cut[0] and
                      doc_cut[0] not in tmp_result and doc_cut[0] not in stopws and doc_cut[0]!=''):
                    tmp_result = [tmp_result, doc_cut[0]]  # 由一个专业名词和一个文章关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(doc_cut) > 1 and tmp_result != doc_cut[1] and tmp_result not in doc_cut[1] and
                      doc_cut[1] not in tmp_result and doc_cut[1] not in stopws and doc_cut[1]!=''):
                    tmp_result = [tmp_result, doc_cut[1]]  # 由一个专业名词和一个文章关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                else:
                    tmp_result = [tmp_result, "朱国杰"]  # 由一个专业名词和特殊词组成，这种情况是为了防止特殊情况
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
            if flag_pro == 0 and flag == 0 and tmp_result == '':  # 说明没找到专业名词，则找关键词中的两个
                if (len(title_cut_1) > 1 and len(title_cut_1[0]) > 1 and len(title_cut_1[1]) > 1
                        and title_cut_1[0] not in stopws and title_cut_1[1] not in stopws):  # 大于1是说明大于一个字符，防止出现一个字母的情况
                    tmp_result = [title_cut_1[0], title_cut_1[1]]  # 由两个题目关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0])> 1 and len(doc_cut)>0 and len(doc_cut[0])>1
                      and title_cut_1[0] not in doc_cut[0] and
                      title_cut_1[0] not in doc_cut[0]):
                    if title_cut_1[0] not in stopws and doc_cut[0] not in stopws:
                        tmp_result = [title_cut_1[0], doc_cut[0]]   # 由一个题目关键词和一个文章关键词组成
                        print(tmp_result)
                        count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                        flag = 1
                    elif title_cut_1[0] in stopws and doc_cut[0] not in stopws:
                        if len(title_cut_1) > 1 and len(title_cut_1[1]) > 1 and title_cut_1[1] not in stopws and doc_cut[0] not in stopws \
                                and title_cut_1[1] not in doc_cut[0] and title_cut_1[1] not in doc_cut[0]:
                            tmp_result = [title_cut_1[1], doc_cut[0]]   # 由一个题目关键词和一个文章关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                        elif len(doc_cut) > 1 and doc_cut[0] not in stopws and doc_cut[1] not in stopws and \
                                doc_cut[0] not in doc_cut[1] and doc_cut[1] not in doc_cut[0]:
                            tmp_result = [doc_cut[0], doc_cut[1]]  # 由两个文章关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                        elif len(doc_cut) > 2 and doc_cut[1] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[1] not in doc_cut[2] and doc_cut[1] not in doc_cut[2]:
                            tmp_result = [doc_cut[1], doc_cut[2]]   # 由两个文章关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                        elif len(doc_cut) > 2 and doc_cut[0] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[0] not in doc_cut[2] and doc_cut[2] not in doc_cut[0]:
                            tmp_result = [doc_cut[0], doc_cut[2]]   # 由两个文章关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                    elif title_cut_1[0] not in stopws and doc_cut[0] in stopws:
                        if len(doc_cut)>1 and doc_cut[1] not in stopws and title_cut_1[0] not in doc_cut[1] and title_cut_1[0] not in doc_cut[1]:
                            tmp_result = [title_cut_1[0], doc_cut[1]]   # 由一个文章关键词和一个题目关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                    elif title_cut_1[0] in stopws and doc_cut[0] in stopws:
                        if len(doc_cut)>1 and doc_cut[1] not in stopws and len(title_cut_1)>1 and title_cut_1[1] not in stopws \
                                and title_cut_1[1] not in doc_cut[1] and title_cut_1[1] not in doc_cut[1]:
                            tmp_result = [title_cut_1[1], doc_cut[1]]   # 由一个文章关键词和一个题目关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                        elif len(title_cut_1)>1 and title_cut_1[1] not in stopws and len(doc_cut)>2\
                            and doc_cut[2] not in stopws and title_cut_1[1] not in doc_cut[2] and title_cut_1[1] not in doc_cut[2]:
                            tmp_result = [title_cut_1[1], doc_cut[2]]  # 由一个文章关键词和一个题目关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                        elif len(doc_cut)>2 and doc_cut[1] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[1] not in doc_cut[2] and doc_cut[1] not in doc_cut[2]:
                            tmp_result = [doc_cut[1], doc_cut[2]]   # 由两个文章关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                        elif len(doc_cut) > 2 and doc_cut[0] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[0] not in doc_cut[2] and doc_cut[2] not in doc_cut[0]:
                            tmp_result = [doc_cut[0], doc_cut[2]]  # 由两个文章关键词组成
                            print(tmp_result)
                            count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                            flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0]) > 1 and len(doc_cut)>1 and
                      title_cut_1[0] not in doc_cut[1] and
                      title_cut_1[0] not in doc_cut[1] and title_cut_1[0] not in stopws and doc_cut[1] not in stopws):
                    tmp_result = [title_cut_1[0], doc_cut[1]]  # 由一个文章关键词和一个题目关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0]) == 0 and len(doc_cut)>1 and len(doc_cut[0])>0 and
                      len(doc_cut[1])>0 and doc_cut[0] not in stopws and doc_cut[1] not in stopws):
                    tmp_result = [doc_cut[0], doc_cut[1]]   # 由两个文章关键词组成
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0]) > 0 and len(doc_cut) >= 1 and len(
                        doc_cut[0]) == 0 and
                      title_cut_1[0] not in stopws):
                    tmp_result = [title_cut_1[0], "朱国杰"]   # 由一个文章关键词和一个特殊名词组成，这是特殊情况，比较少
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1
                else:  # 比如题目为：一口气看完，太荡气回肠了
                    tmp_result = ["朱国杰", "朱国杰"]   # 由两个特殊名词组成，这是特殊情况，比较少
                    print(tmp_result)
                    count = count + len(set(result[:2]) & set(tmp_result[:2]))  # 此时已经有两个关键词可以比对了
                    flag = 1

    print(count)

    if len(tmp_result) > 1:
        label1.append(tmp_result[0])
        label2.append(tmp_result[1])
    elif len(tmp_result) == 1:
        label1.append(tmp_result[0])
        label2.append(tmp_result[0])
    else:
        label1.append('')
        label2.append('')

result = pd.DataFrame()

id = train_title_doc['id'].unique()

result['id'] = list(id)
result['label1'] = label1
result['label1'] = result['label1'].replace('', 'nan')
result['label2'] = label2
result['label2'] = result['label2'].replace('', 'nan')

# 策略 extract_tags
test_title_doc['title_cut'] = test_title_doc['title'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890', str(x))))

test_title_doc['title_cut_1'] = test_title_doc['title_cut'].apply(lambda x:','.join(jieba.analyse.extract_tags(str(x),topK = 5,allowPOS=('nt','ns','nz',"n", "vn", "v", "nr"))))
test_title_doc['title_cut_2'] = test_title_doc['title_cut'].apply(lambda x:HanLP.parseDependency(x))
# 分割标题各词的词性
test_title_doc['title_cut_3'] = test_title_doc['title_cut'].apply(lambda x:HanLP.segment(x))

# 去除文章的数字，数字没有意义，单纯一个数字，不能达到对文章内容的区分,这里用了doc文章
test_title_doc['doc_cut'] = test_title_doc['doc'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890', str(x))))
# 策略 extract_tags 直接利用jieba的提取主题词的工具
test_title_doc['doc_cut'] = test_title_doc['doc_cut'].apply(lambda x:','.join(jieba.analyse.textrank(str(x),topK = 5,allowPOS=('nt','ns','nz',"n", "vn", "v", "nr"))))
# 第二规则 提取 《》
test_title_doc['title_regex'] = test_title_doc['title'].apply(lambda x:','.join(re.findall(r"《(.+?)》",str(x))))

# 利用策略 + 规则 查看训练集的准确率
test_offline_result = test_title_doc[['id','id','title_cut_1','title_cut_2','title_cut_3','doc_cut','title_regex']]

label1 = []
label2 = []

for i in test_offline_result.values:
    flag = 0
    flag_name = 0
    flag_pro = 0
    tmp_result = ''
    result = str(i[1]).split(',')
    title_cut_1 = str(i[2]).split(',')
    title_cut_2 = i[3]
    title_cut_3 = i[4]
    doc_cut = str(i[5]).split(',')  # doc_cut列
    title_regex = str(i[6]).split(',')

    print('title_cut_1:', title_cut_1)
    print('doc_cut:', doc_cut)

    # 所有规则整合到一起
    if len(title_regex) >= 2:  # 如果有两个《》
        tmp_result = title_regex
        flag = 1
    elif len(title_regex) == 1 and title_regex[0] != '':  # 如果只有一个《》
        print("title_regex:", title_regex)
        tmp_result = title_regex[0]
        print("title_regex[0]:", title_regex[0])
        for words_title_cut_3 in title_cut_3:  # 遍历看是否有人名
            if flag == 0:
                if str(words_title_cut_3.nature) in ['nr', 'nrf', 'nrj', 'nr2','nz']:  # 先判断真正的人名
                    if len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result and words_title_cut_3.word not in tmp_result \
                            and tmp_result not in words_title_cut_3.word and words_title_cut_3.word not in stopws:
                        tmp_result = [tmp_result, words_title_cut_3.word]  # 由《》和人名组成
                        print(tmp_result)
                        flag = 1
                        break  # 用于跳出for循环
        if flag == 0:  # 如果找不到人名
            for words_title_cut_3 in title_cut_3:  # 遍历寻找专业名词（地名等）
                if flag == 0:
                    if str(words_title_cut_3.nature) in ['ni', 'nic', 'nis', 'nit', 'ns', 'nsf', 'nt', 'ntc', 'ntcb',
                                                         'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
                                                         'nl','nm','nmc','nn','nb','nba','nbc','nbp','nf',
                                                         'ng','nh','nhd','nhm']:
                        if len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                                and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word\
                                and words_title_cut_3.word not in stopws:
                            tmp_result = [tmp_result, words_title_cut_3.word]  # 由《》和专业名词组成
                            print(tmp_result)
                            flag = 1
                            break  # 用于跳出for循环
        if flag == 0:  # 如果找不到人名和其他专业名词，则把提取的题目中的第一个关键字拿来（注意，这个关键字不能和《》内的重复
            if (len(title_cut_1) > 0 and tmp_result != title_cut_1[0] and tmp_result not in title_cut_1[0] and
                    title_cut_1[0] not in tmp_result and title_cut_1[0] not in stopws):
                tmp_result = [tmp_result, title_cut_1[0]]   # 由《》和题目关键字组成
                print(tmp_result)
                flag = 1
            elif (len(title_cut_1) > 1 and tmp_result != title_cut_1[1] and tmp_result not in title_cut_1[1] and
                  title_cut_1[1] not in tmp_result and title_cut_1[1] not in stopws):
                tmp_result = [tmp_result, title_cut_1[1]]   # 由《》和题目关键字组成
                print(tmp_result)
                flag = 1
            elif (len(title_cut_1) > 2 and tmp_result != title_cut_1[2] and tmp_result not in title_cut_1[2] and
                  title_cut_1[2] not in tmp_result and title_cut_1[2] not in stopws):
                tmp_result = [tmp_result, title_cut_1[2]]  # 由《》和题目关键字组成
                print(tmp_result)
                flag = 1
            elif (len(doc_cut) > 0 and tmp_result != doc_cut[0] and tmp_result not in doc_cut[0] and
                  doc_cut[0] not in tmp_result and doc_cut[0] not in stopws):
                tmp_result = [tmp_result, doc_cut[0]]    # 由《》和文章关键字组成
                print(tmp_result)
                flag = 1
            elif (len(doc_cut) > 1 and tmp_result != doc_cut[1] and tmp_result not in doc_cut[1] and
                  doc_cut[1] not in tmp_result and doc_cut[1] not in stopws):
                tmp_result = [tmp_result, doc_cut[1]]   # 由《》和文章关键字组成
                print(tmp_result)
                flag = 1
            elif (len(doc_cut) > 2 and tmp_result != doc_cut[2] and tmp_result not in doc_cut[2] and
                  doc_cut[2] not in tmp_result and doc_cut[2] not in stopws):
                tmp_result = [tmp_result, doc_cut[2]]   # 由《》和文章关键字组成
                print(tmp_result)
                flag = 1
            else:
                tmp_result = [tmp_result, "朱国杰"]   # 由《》和随便取的名字组成，因为可能存在很特殊的情况，所以加了这个
                print(tmp_result)
                flag = 1
    elif len(title_regex) == 1 and flag == 0 and title_regex[0] == '':  # 没有《》
        tmp_result = ""  # 其实上一个if应该是能保证有两个关键词的，以防万一，这里又重新初始化
        for words_title_cut_3 in title_cut_3:
            if flag == 0:
                if str(words_title_cut_3.nature) in ['nr', 'nrf', 'nrj', 'nr2','nz']:
                    if tmp_result == '' and len(words_title_cut_3.word) > 1 and words_title_cut_3.word not in stopws:  # 找到一个人名
                        tmp_result = words_title_cut_3.word
                        print('words_title_cut_3.word:',words_title_cut_3.word)
                        flag_name = 1  # 标志着找到一个人名
                    elif tmp_result != '' and len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                            and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word \
                            and words_title_cut_3.word not in stopws:
                        tmp_result = [tmp_result, words_title_cut_3.word]   # 由两个人名组成
                        print(tmp_result)
                        flag = 1
                        flag_name = 2  # 标志着找到两个人名
                        break  # 用于跳出for循环
        if flag_name == 1 and flag == 0 and tmp_result != '':  # 能到这个if,说明遍历过了，但是只找到一个人名
            for words_title_cut_3 in title_cut_3:  # 在出现了一个人名的情况下，重新遍历,看是否有专业名词
                if flag == 0:
                    if str(words_title_cut_3.nature) in ['ni', 'nic', 'nis', 'nit', 'ns', 'nsf', 'nt', 'ntc', 'ntcb',
                                                         'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
                                                         'nl','nm','nmc','nn','nb','nba','nbc','nbp','nf',
                                                         'ng','nh','nhd','nhm']:
                        if len(words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                                and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word \
                                and words_title_cut_3.word not in stopws:
                            tmp_result = [tmp_result, words_title_cut_3.word]  # 由一个人名和一个专业名词组成
                            print(tmp_result)
                            flag = 1
                            break  # 用于跳出for循环
            if flag == 0:  # 说明没有专业名词，则取题目关键词中的第一个
                if (len(title_cut_1) > 0 and tmp_result != title_cut_1[0] and tmp_result not in title_cut_1[0] and
                        title_cut_1[0] not in tmp_result and title_cut_1[0] not in stopws):
                    tmp_result = [tmp_result, title_cut_1[0]]   # 由一个人名和题目关键字组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) > 1 and tmp_result != title_cut_1[1] and tmp_result not in title_cut_1[1] and
                      title_cut_1[1] not in tmp_result and title_cut_1[1] not in stopws):
                    tmp_result = [tmp_result, title_cut_1[1]]  # 由一个人名和题目关键字组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) == 1 and title_cut_1[0] == tmp_result and len(doc_cut)>0 and len(doc_cut[0])>1
                      and tmp_result != doc_cut[0] and tmp_result not in doc_cut[0] and
                      doc_cut[0] not in tmp_result and doc_cut[0] not in stopws):
                    tmp_result = [tmp_result, doc_cut[0]]   # 由一个人名和文章关键字组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) == 1 and title_cut_1[0] == tmp_result and len(doc_cut)>1 and
                      tmp_result != doc_cut[1] and tmp_result not in doc_cut[1] and
                      doc_cut[1] not in tmp_result and doc_cut[1] not in stopws):
                    tmp_result = [tmp_result, doc_cut[1]]   # 由一个人名和文章关键字组成
                    print(tmp_result)
                    flag = 1
                else:  # 出现一些特殊情况，比如人名或者专业名词很长，涵盖了几个比较重要的词，或者正文doc很短很短，这里想改还可以改，不过情况比较多！！！
                    tmp_result = [tmp_result, doc_cut[0]]   # 这里的doc_cut[0]有可能是''，# 由一个人名和文章关键字组成
                    print(tmp_result)
                    flag = 1
        if flag_name == 0 and flag == 0:  # 说明没有人名，则重新挑专业名词
            tmp_result = ""
            for words_title_cut_3 in title_cut_3:
                if flag == 0:
                    if str(words_title_cut_3.nature) in ['ni', 'nic', 'nis', 'nit', 'ns', 'nsf', 'nt', 'ntc', 'ntcb',
                                                         'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
                                                         'nl','nm','nmc','nn','nb','nba','nbc','nbp','nf',
                                                         'ng','nh','nhd','nhm']:
                        if tmp_result == '' and len(words_title_cut_3.word) > 1 and words_title_cut_3.word not in stopws:
                            tmp_result = words_title_cut_3.word
                            flag_pro = 1  # 说明找到一个专业名词
                        elif tmp_result != '' and len(
                                words_title_cut_3.word) > 1 and words_title_cut_3.word != tmp_result \
                                and words_title_cut_3.word not in tmp_result and tmp_result not in words_title_cut_3.word \
                                and words_title_cut_3.word not in stopws:
                            tmp_result = [tmp_result, words_title_cut_3.word]  # 由两个专业名词组成
                            print(tmp_result)
                            flag = 1
                            flag_pro = 2  # 说明找到两个专业名词
                            break  # 用于跳出for循环
            if flag_pro == 1 and flag == 0 and tmp_result != '':  # 说明只找到一个专业名词，则再取关键词中的一个
                if (len(title_cut_1) > 0 and tmp_result != title_cut_1[0] and tmp_result not in title_cut_1[0] and
                        title_cut_1[0] not in tmp_result and title_cut_1[0] not in stopws and title_cut_1[0]!=''):
                    tmp_result = [tmp_result, title_cut_1[0]]    # 由一个专业名词和一个题目关键词组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) > 1 and tmp_result != title_cut_1[1] and tmp_result not in title_cut_1[1] and
                      title_cut_1[1] not in tmp_result and title_cut_1[1] not in stopws and title_cut_1[1]!=''):
                    tmp_result = [tmp_result, title_cut_1[1]]  # 由一个专业名词和一个题目关键词组成
                    print(tmp_result)
                    flag = 1
                elif (len(doc_cut) > 0 and tmp_result != doc_cut[0] and tmp_result not in doc_cut[0] and
                      doc_cut[0] not in tmp_result and doc_cut[0] not in stopws and doc_cut[0]!=''):
                    tmp_result = [tmp_result, doc_cut[0]]  # 由一个专业名词和一个文章关键词组成
                    print(tmp_result)
                    flag = 1
                elif (len(doc_cut) > 1 and tmp_result != doc_cut[1] and tmp_result not in doc_cut[1] and
                      doc_cut[1] not in tmp_result and doc_cut[1] not in stopws and doc_cut[1]!=''):
                    tmp_result = [tmp_result, doc_cut[1]]  # 由一个专业名词和一个文章关键词组成
                    print(tmp_result)
                    flag = 1
                else:
                    tmp_result = [tmp_result, "朱国杰"]  # 由一个专业名词和特殊词组成，这种情况是为了防止特殊情况
                    print(tmp_result)
                    flag = 1
            if flag_pro == 0 and flag == 0 and tmp_result == '':  # 说明没找到专业名词，则找关键词中的两个
                if (len(title_cut_1) > 1 and len(title_cut_1[0]) > 1 and len(title_cut_1[1]) > 1
                        and title_cut_1[0] not in stopws and title_cut_1[1] not in stopws):  # 大于1是说明大于一个字符，防止出现一个字母的情况
                    tmp_result = [title_cut_1[0], title_cut_1[1]]  # 由两个题目关键词组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0])> 1 and len(doc_cut)>0 and len(doc_cut[0])>1
                      and title_cut_1[0] not in doc_cut[0] and
                      title_cut_1[0] not in doc_cut[0]):
                    if title_cut_1[0] not in stopws and doc_cut[0] not in stopws:
                        tmp_result = [title_cut_1[0], doc_cut[0]]   # 由一个题目关键词和一个文章关键词组成
                        print(tmp_result)
                        flag = 1
                    elif title_cut_1[0] in stopws and doc_cut[0] not in stopws:
                        if len(title_cut_1) > 1 and len(title_cut_1[1]) > 1 and title_cut_1[1] not in stopws and doc_cut[0] not in stopws \
                                and title_cut_1[1] not in doc_cut[0] and title_cut_1[1] not in doc_cut[0]:
                            tmp_result = [title_cut_1[1], doc_cut[0]]   # 由一个题目关键词和一个文章关键词组成
                            print(tmp_result)
                            flag = 1
                        elif len(doc_cut) > 1 and doc_cut[0] not in stopws and doc_cut[1] not in stopws and \
                                doc_cut[0] not in doc_cut[1] and doc_cut[1] not in doc_cut[0]:
                            tmp_result = [doc_cut[0], doc_cut[1]]  # 由两个文章关键词组成
                            print(tmp_result)
                            flag = 1
                        elif len(doc_cut) > 2 and doc_cut[1] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[1] not in doc_cut[2] and doc_cut[1] not in doc_cut[2]:
                            tmp_result = [doc_cut[1], doc_cut[2]]   # 由两个文章关键词组成
                            print(tmp_result)
                            flag = 1
                        elif len(doc_cut) > 2 and doc_cut[0] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[0] not in doc_cut[2] and doc_cut[2] not in doc_cut[0]:
                            tmp_result = [doc_cut[0], doc_cut[2]]   # 由两个文章关键词组成
                            print(tmp_result)
                            flag = 1
                    elif title_cut_1[0] not in stopws and doc_cut[0] in stopws:
                        if len(doc_cut)>1 and doc_cut[1] not in stopws and title_cut_1[0] not in doc_cut[1] and title_cut_1[0] not in doc_cut[1]:
                            tmp_result = [title_cut_1[0], doc_cut[1]]   # 由一个文章关键词和一个题目关键词组成
                            print(tmp_result)
                            flag = 1
                    elif title_cut_1[0] in stopws and doc_cut[0] in stopws:
                        if len(doc_cut)>1 and doc_cut[1] not in stopws and len(title_cut_1)>1 and title_cut_1[1] not in stopws \
                                and title_cut_1[1] not in doc_cut[1] and title_cut_1[1] not in doc_cut[1]:
                            tmp_result = [title_cut_1[1], doc_cut[1]]   # 由一个文章关键词和一个题目关键词组成
                            print(tmp_result)
                            flag = 1
                        elif len(title_cut_1)>1 and title_cut_1[1] not in stopws and len(doc_cut)>2\
                            and doc_cut[2] not in stopws and title_cut_1[1] not in doc_cut[2] and title_cut_1[1] not in doc_cut[2]:
                            tmp_result = [title_cut_1[1], doc_cut[2]]  # 由一个文章关键词和一个题目关键词组成
                            print(tmp_result)
                            flag = 1
                        elif len(doc_cut)>2 and doc_cut[1] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[1] not in doc_cut[2] and doc_cut[1] not in doc_cut[2]:
                            tmp_result = [doc_cut[1], doc_cut[2]]   # 由两个文章关键词组成
                            print(tmp_result)
                            flag = 1
                        elif len(doc_cut) > 2 and doc_cut[0] not in stopws and doc_cut[2] not in stopws and \
                                doc_cut[0] not in doc_cut[2] and doc_cut[2] not in doc_cut[0]:
                            tmp_result = [doc_cut[0], doc_cut[2]]  # 由两个文章关键词组成
                            print(tmp_result)
                            flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0]) > 1 and len(doc_cut)>1 and
                      title_cut_1[0] not in doc_cut[1] and
                      title_cut_1[0] not in doc_cut[1] and title_cut_1[0] not in stopws and doc_cut[1] not in stopws):
                    tmp_result = [title_cut_1[0], doc_cut[1]]  # 由一个文章关键词和一个题目关键词组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0]) == 0 and len(doc_cut)>1 and len(doc_cut[0])>0 and
                      len(doc_cut[1])>0 and doc_cut[0] not in stopws and doc_cut[1] not in stopws):
                    tmp_result = [doc_cut[0], doc_cut[1]]   # 由两个文章关键词组成
                    print(tmp_result)
                    flag = 1
                elif (len(title_cut_1) >= 1 and len(title_cut_1[0]) > 0 and len(doc_cut) >= 1 and len(
                        doc_cut[0]) == 0 and
                      title_cut_1[0] not in stopws):
                    tmp_result = [title_cut_1[0], "朱国杰"]   # 由一个文章关键词和一个特殊名词组成，这是特殊情况，比较少
                    print(tmp_result)
                    flag = 1
                else:  # 比如题目为：一口气看完，太荡气回肠了
                    tmp_result = ["朱国杰", "朱国杰"]   # 由两个特殊名词组成，这是特殊情况，比较少
                    print(tmp_result)
                    flag = 1

    if len(tmp_result) > 1:
        label1.append(tmp_result[0])
        label2.append(tmp_result[1])
    elif len(tmp_result) == 1:
        label1.append(tmp_result[0])
        label2.append(tmp_result[0])
    else:
        label1.append('')
        label2.append('')

result = pd.DataFrame()

id = test_title_doc['id'].unique()

result['id'] = list(id)
result['label1'] = label1
result['label1'] = result['label1'].replace('','nan')
result['label2'] = label2
result['label2'] = result['label2'].replace('','nan')

print(result)
result.to_csv('../result/jieb_ruler_result_1011_2.csv',index=None)