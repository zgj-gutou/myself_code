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
from itertools import combinations

start_time = time.time()
train = pd.read_csv('/data/atec_anti_fraud_train.csv')
test_x = pd.read_csv('/data/atec_anti_fraud_test_b.csv')
test2 = pd.read_csv('/data/test2.csv')

i = 0

print("-----------------------------------------------------")
# print(train['date'].value_counts())
# print(train.iloc[:,:50].describe())
print("load data prepared!")

print(train.label.value_counts())
train.label = train.label.replace(-1,1)
print(train.label.value_counts())

train = train[(train.label == 1) | (train.label == 0)]
train['year'] = train['date']/10000
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
train_na=(temp_A==-1).sum(axis=1)

#---val_na--------------------------------------------------------------------
temp_C=validation_x.copy()
temp_C=temp_C.fillna(-1)
val_na=(temp_C==-1).sum(axis=1)

#---test_na--------------------------------------------------------------------
temp_B=test_x.copy()
temp_B=temp_B.fillna(-1)
test_na=(temp_B==-1).sum(axis=1)

train_x_five['na']=train_na
validation_x['na']=val_na
test_x['na']=test_na

del temp_A,temp_B,temp_C,train_na,val_na,test_na


feature=[item for item in train_x_five.columns.tolist() if item not in ['date']]

feature_unique1=[train_x_five[item].nunique()  for item in feature]
feature_unique=pd.DataFrame()
feature_unique['feature_name']=list(feature)
feature_unique['uni']=feature_unique1
feature_unique=feature_unique.sort_values(by='uni',ascending=False)

cat_feature=list(feature_unique.loc[feature_unique['uni']<=10,'feature_name'])
num_feature=[item for item in feature if item not in cat_feature]


cat_count_features = []
for c in cat_feature:
    d = pd.concat([train_x_five[c],validation_x[c],test_x[c]]).value_counts().to_dict()
    train_x_five['%s_count'%c] =train_x_five[c].apply(lambda x:d.get(x,0))
    validation_x['%s_count'%c] = validation_x[c].apply(lambda x:d.get(x,0))
    test_x['%s_count'%c] = test_x[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)
    
train_x_five=train_x_five.drop('date',axis=1)
validation_x=validation_x.drop('date',axis=1)
test_x=test_x.drop('date',axis=1)

train_x_five=train_x_five.fillna(-1)
validation_x=validation_x.fillna(-1)
test_x=test_x.fillna(-1)

#加载特征重要性特征前200维
select_features = list(pd.read_csv('/feature_importance.csv')['feature'])[:200]
#combine_fe = list(pd.read_csv('D:/mayi/feature_imp/feature_importance.csv')['feature'])[:11]
#combine_fe.remove('day')


train_x_five=train_x_five[select_features]
validation_x=validation_x[select_features]
test_x=test_x[select_features]

'''
def interaction_features(train,val,test, fea1, fea2):
    
    train[fea1+'*'+fea2] = train[fea1] * train[fea2]
    train[fea1+'/'+fea2] = train[fea1] / train[fea2]
   
    val[fea1+'*'+fea2] = val[fea1] * val[fea2]
    val[fea1+'/'+fea2] = val[fea1] / val[fea2]
    
    test[fea1+'*'+fea2] = test[fea1] * test[fea2]
    test[fea1+'/'+fea2] = test[fea1] / test[fea2]


    return train,val,test


for e, (x, y) in enumerate(combinations(combine_fe, 2)):
    train_x_five,validation_x,test_x = interaction_features(train_x_five,validation_x,test_x,x, y)
'''

print(train_x_five.shape)
print(validation_x.shape)
print(test_x.shape)

#-----------------------------------模型-----------------------------------------------------------------
from sklearn.metrics import roc_curve
def score(y,pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    score=0.4*tpr[np.where(fpr >=0.001)[0][0]]+0.3*tpr[np.where(fpr >=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]]
    return score

#-----------------------------------lgb--------------------------
def LGB_predict(train_x,train_y,validation_x,validation_y,test_x,test2):
    print("LGB test")
    global i

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.005, min_child_weight=20, random_state=27, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(validation_x,validation_y)], eval_metric='auc', early_stopping_rounds=100)
    print(clf)
    print('n_feature:',clf.n_features_)  # 可以看到模型需要几个输入特征
    print('self:',test_x.shape)  # 根据validation_x的列可以看出有几个输入特征，也可以直接写为validation_x[1]
    # 模型需要的输入特征要和实际输入特征一样
    print('---------------------------------------------------')
    a = score(train_y,clf.predict_proba(train_x)[:, 1])
    print(a)
    print('----------------------------------------------------')
    test2['score'] = clf.predict_proba(test_x)[:, 1]
    test2['score'] = test2['score'].apply(lambda x: float('%.4f' % x))
    test2.to_csv('/lgb_4338.csv', index=False)
    print(i)
    print('over')
    i = i+1
    return clf
	
#-----------------------------------------------------------xgb----------------------------------------------------	
def XGB_predict(train_x,train_y,validation_x,validation_y,test_x,test2):
    print("XGB test")
    global i
    
    clf=xgb.XGBClassifier( booster='gbtree',objective='binary:logistic',
                             scale_pos_weight=1,colsample_bytree=0.7, gamma=0.0, 
                             learning_rate=0.005, max_depth=5,
                             min_child_weight=20, n_estimators=2000,
                             subsample=0.7, silent=True,
                             nthread = -1,random_state=27)
    clf.fit(train_x, train_y)
    print(clf)
    #print('n_feature:',clf.n_features_)  # 可以看到模型需要几个输入特征
    print('self:',test_x.shape)  # 根据validation_x的列可以看出有几个输入特征，也可以直接写为validation_x[1]
    # 模型需要的输入特征要和实际输入特征一样
    print('---------------------------------------------------')
    a = score(train_y,clf.predict_proba(train_x)[:, 1])
    print(a)
    print('----------------------------------------------------')
    test2['score'] = clf.predict_proba(test_x)[:, 1]
    test2['score'] = test2['score'].apply(lambda x: float('%.4f' % x))
    test2.to_csv('/xgb_4312.csv', index=False)
    print(i)
    print('over')
    i = i+1
    return clf
model_lgb = LGB_predict(train_x_five, train_y_five, validation_x,validation_y,test_x, test2)
model_xgb = XGB_predict(train_x_five, train_y_five, validation_x,validation_y,test_x, test2)
#del train_x_five, train_y_five

end_time = time.time()
print('time used:',end_time-start_time)


#-----------------模型融合-------------------------------------------------------------
# rank_融合  res=0.65*lgb_rank+0.35*xgb_rank    res_4437
import pandas as pd

path=''
xgb=pd.read_csv(path+'xgb_4312.csv')
lgb=pd.read_csv(path+'lgb_4338.csv')


def get_rank(x):
    return pd.Series(x).rank(pct=True).values

res=pd.DataFrame({'id': xgb['id'], 'score':get_rank(xgb['score']) * 0.35 + get_rank(lgb['score']) * 0.65})
res.to_csv(path+"ronghe.csv", index=None)
print(res.head(5))


