import warnings
import sys
# sys.path.append("/notebook/shared/extra/")
# warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import os
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 

start_time = time.time()
train = pd.read_csv('D:/mayi/data/atec_anti_fraud_train.csv')
test_x = pd.read_csv('D:/mayi/data/atec_anti_fraud_test_b.csv')
test2 = pd.read_csv('D:/mayi/data/test2.csv')

i = 0

print("-----------------------------------------------------")
# print(train['date'].value_counts())
# print(train.iloc[:,:50].describe())
print("load data prepared!")

print(train.label.value_counts())
train.label = train.label.replace(-1,1)
print(train.label.value_counts())

train = train[(train.label == 1) | (train.label == 0)]
train['year'] = train['date']/10000     # 20171103
train['month'] = train['date']%10000/100
train['day'] = train['date']%1000000

test_x['year'] = test_x['date']/10000
test_x['month'] = test_x['date']%10000/100
test_x['day'] = test_x['date']%1000000

train_train = train[(train.date <= 20171007) ]   # 训练集，原先训练集的一部分
train_validation = train[(train.date > 20171007) ]   # 验证集，原先训练集的一部分

train_train_1th_fold = train_train[(train.date <= 20170912) ]
train_train_2th_fold = train_train[(train.date <= 20170918) & (train.date > 20170912) ]
train_train_3th_fold = train_train[(train.date <= 20170924) & (train.date > 20170918) ]
train_train_4th_fold = train_train[(train.date <= 20170930) & (train.date > 20170924) ]
train_train_5th_fold = train_train[(train.date <= 20171007) & (train.date > 20170930) ]


frames5 = [train_train_2th_fold, train_train_3th_fold,train_train_4th_fold,train_train_5th_fold]
train_train_five = pd.concat(frames5).reset_index(drop=True)


train_y_five = train_train_five.label    # 训练集的label
train_x_five = train_train_five.drop(['label','id','year','month'],axis =1)  # 训练集的特征数据

train_y = train_train.label    # 训练集的label
train_x = train_train.drop(['label','id','year','month'],axis =1)  # 训练集的特征数据
validation_y = train_validation.label   # 验证集的label
validation_x = train_validation.drop(['label','id','year','month'],axis =1)  # 验证集的特征数

test_x = test_x.drop(['id','year','month'],axis =1)  # 测试集的特征数

del train_train_5th_fold,frames5,train

#---train_na--------------------------------------------------------------------
temp_A=train_x_five.copy()
temp_A=temp_A.fillna(-1)
train_na=(temp_A==-1).sum(axis=1)    # 空值个数

#---val_na--------------------------------------------------------------------
temp_C=validation_x.copy()
temp_C=temp_C.fillna(-1)
val_na=(temp_C==-1).sum(axis=1)     # 空值个数

#---test_na--------------------------------------------------------------------
temp_B=test_x.copy()
temp_B=temp_B.fillna(-1)
test_na=(temp_B==-1).sum(axis=1)    # 空值个数

train_x_five['na']=train_na
validation_x['na']=val_na
test_x['na']=test_na

del temp_A,temp_B,temp_C,train_na,val_na,test_na


feature=[item for item in train_x_five.columns.tolist() if item not in ['date']]    # 列出除了date以外的特征

feature_unique1=[train_x_five[item].nunique()  for item in feature]  # unique()看看该列有多少不同的元素,统计不同元素的个数
feature_unique=pd.DataFrame()
feature_unique['feature_name']=list(feature)
feature_unique['uni']=feature_unique1
feature_unique=feature_unique.sort_values(by='uni',ascending=False)   # 根据不同元素个数降序排列

cat_feature=list(feature_unique.loc[feature_unique['uni']<=10,'feature_name'])   # 把不同元素个数小于10个的当做类别特征
num_feature=[item for item in feature if item not in cat_feature]


cat_count_features = []
for c in cat_feature:
    d = pd.concat([train_x_five[c],validation_x[c],test_x[c]]).value_counts().to_dict()  # 统计每个类别变量的名称和个数为一个字典
    train_x_five['%s_count'%c] =train_x_five[c].apply(lambda x:d.get(x,0))   # c类别变量在train_x_five中的个数，如果没有则为0个
    validation_x['%s_count'%c] = validation_x[c].apply(lambda x:d.get(x,0))
    test_x['%s_count'%c] = test_x[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)
    
train_x_five=train_x_five.drop('date',axis=1)   # 删除date特征
validation_x=validation_x.drop('date',axis=1)
test_x=test_x.drop('date',axis=1)

train_x_five=train_x_five.fillna(-1)  # 对空值填-1
validation_x=validation_x.fillna(-1)
test_x=test_x.fillna(-1)


model_xgb= xgb.XGBClassifier(n_estimators=100,max_depth=5,random_state=0)
model_xgb.fit(train_x_five,train_y_five)

feature=list(train_x_five.columns)
feature_importance = pd.DataFrame({'feature':feature,'importance':model_xgb.feature_importances_})  # 对每个特征的重要性做一个dataframe
feature_importance = feature_importance.sort_values('importance',ascending=False)  # 对重要性降序
feature_importance.to_csv('/feature_importance.csv',index=None)
