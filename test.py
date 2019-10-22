# -*- coding: UTF-8 -*-

from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
from scipy.stats import rankdata
from scipy import sparse
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import datetime
import time
import os

path = '.'

warnings.filterwarnings("ignore")


def show_NaN(df):
    NAs = df.isnull().sum()
    print(NAs[NAs > 0])


def get_ctr_matrix(data, src, dest, others=[]):

    temp = data.loc[data['label'] != -1, [src, dest, 'label']]
    
    src_gb = temp.groupby([src])
    src_frame = src_gb.agg({'label': 'sum'}).rename(columns={'label': 'src_sum'}).join(
        src_gb.agg({'label': 'count'}).rename(columns={'label': 'src_count'})).reset_index()
    del src_gb

    dest_gb = temp.groupby([src, dest])
    dest_frame = dest_gb.agg({'label': 'sum'}).rename(columns={'label': 'dest_sum'}).join(
        dest_gb.agg({'label': 'count'}).rename(columns={'label': 'dest_count'})).reset_index()
    del dest_gb

    df = pd.merge(src_frame, dest_frame, on=[src], how='right')
    del temp, src_frame, dest_frame

    df['rate'] = (df['src_sum'] + df['dest_sum']) / (df['src_count'] + df['dest_count'])
    df.drop(['src_sum', 'src_count', 'dest_sum', 'dest_count'], axis=1, inplace=True)
    data = pd.merge(data, df, on=[src, dest], how='left').fillna(0)
    del df

    src_hot = pd.get_dummies(data[src])
    src_hot.columns = ['{}_{}_{}_ctr'.format(src, dest, h) for h in src_hot.columns]
    for h in src_hot.columns:
        src_hot[h] = (src_hot[h] * data['rate']).astype('float32')
    data.drop(['rate'], axis=1, inplace=True)
    data = pd.concat([data, src_hot], axis=1)
    src_ctr_cols = list(src_hot.columns)

    if len(others) > 0:
        src_hot = pd.concat([data[others], src_hot], axis=1)
        for other in others:
            other_mean = src_hot[[other] + src_ctr_cols].groupby(other).mean().reset_index()
            other_mean_cols = ['{}_mean_{}'.format(i, other) for i in src_ctr_cols]
            other_mean.columns = [other] + other_mean_cols
            src_hot = pd.merge(src_hot, other_mean, on=[other], how='left')
            src_hot.drop([other], axis=1, inplace=True)

    return sparse.csr_matrix(src_hot[data['label'] != -1], dtype='float32'), sparse.csr_matrix(src_hot[data['label'] == -1], dtype='float32')


def get_user_preference_feature(data, prop_col, srcs):

    def __get_one_user_preference_feature(data, prop_cols, *src_cols):
        vector_cols = []
        dest_cols = prop_cols
        for src in src_cols:
            dest_mean = data[[src] + dest_cols].groupby([src]).mean().reset_index()
            dest_cols = ['{}_mean_{}'.format(dest, src) for dest in dest_cols]
            dest_mean.columns = [src] + dest_cols
            data = pd.merge(data, dest_mean, on=[src], how='left')
            vector_cols += dest_cols
        return data, vector_cols

    prop_hot = pd.get_dummies(data[prop_col], dtype='bool')
    prop_hot.columns = ['{}_{}'.format(prop_col, i) for i in prop_hot.columns]
    data = pd.concat([data, prop_hot], axis=1)
    prop_cols = list(prop_hot.columns)

    vector_cols = []
    for (src1, src2) in srcs:
        data, cols = __get_one_user_preference_feature(data, prop_cols, src1, src2)
        vector_cols += cols
    
    data.drop(prop_cols, axis=1, inplace=True)
    return data, vector_cols


df = pd.read_table(path + '/Data/round1_iflyad_train.txt')
df['day'] = df['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
df['hour'] = df['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

train = df[df.day != 2]
test = df[df.day == 2]
y_label = test.click.values.copy()
test['click'] = -1
print('dest', y_label.sum() / len(y_label))
data = pd.concat([train, test], axis=0, ignore_index=True)

data = data.fillna(-1)
data['label'] = data.click.astype(int)
del data['click']

data['advert_industry_inner_1'] = data['advert_industry_inner'].str.split('_', expand=True)[0]

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature
for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique())))).astype('int32')
print('label encoder done!')

#------------------ ctr feature -------------------
ctr_list = [
    
    # # ('creative_height', 'inner_slot_id'), # logloss 0.4238442664489534
    # # ('advert_id', 'creative_id'),         # logloss 0.4243040318885814
    # # ('campaign_id', 'creative_id'),       # logloss 0.42430706796201295 删原type以后
    # # ('nnt', 'app_cate_id'),               # logloss 0.42393648608335555

    # # ad
    ('creative_width', 'app_id', ['osv']), # logloss 0.42571373442749605 # model logloss 0.42507915435465377 # osv logloss 0.4249699664512641
    ('creative_width', 'inner_slot_id', []),     # logloss 0.42380610799354385
    # logloss 0.4251582631742659
    # logloss 0.4257414364203091 not add 3

    # ('advert_id', 'app_id'),                 # logloss 0.4242214362090032
    # ('advert_id', 'inner_slot_id'),          # logloss 0.42429694231070747
    # ('campaign_id', 'inner_slot_id'),        # logloss 0.4242488450116603
    # ('campaign_id', 'app_id'),               # logloss 0.42425235520405014
    # ('campaign_id', 'app_cate_id'),          # logloss 0.4242960748977619
    # # app
    # ('app_cate_id', 'creative_id'),          # logloss 0.4242633651052074
    # # user
    # ('nnt', 'creative_id', ['province']),
]
# logloss 0.4260458328562308
# logloss 0.4252385884354041]

ctr_feature = []
for (src, dest, others) in ctr_list:
    ctr_feature += ['{}_{}_others'.format(src, dest)]

#------------------------------------------------
cate_feature = origin_cate_list
num_feature = ['creative_width', 'creative_height', 'hour']

feature = cate_feature + num_feature + ctr_feature
print(len(feature), feature)

predict = data[data.label == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

ctr_train_csr = sparse.csr_matrix((len(train_x), 0))
ctr_predict_csr = sparse.csr_matrix((len(predict), 0))
for (src, dest, others) in ctr_list:
    ctr_t, ctr_p = get_ctr_matrix(data, src, dest, others)
    ctr_train_csr = sparse.hstack((ctr_train_csr, ctr_t), 'csr', 'float32')
    ctr_predict_csr = sparse.hstack((ctr_predict_csr, ctr_p), 'csr', 'float32')
    print(src, dest, 'over')
print('ctr feature prepared !')

num_train_csr = sparse.hstack((ctr_train_csr, sparse.csr_matrix(train_x[num_feature])), 'csr', 'float32')
num_predict_csr = sparse.hstack((ctr_predict_csr, sparse.csr_matrix(predict[num_feature])), 'csr', 'float32')
print('number feature prepared !')

# 默认加载 如果 增加了cate类别特征 请改成false重新生成
if os.path.exists(path + '/feature/test_train_csr.npz') and True:
    print('load_csr---------')
    base_train_csr = sparse.load_npz(path + '/feature/test_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz(path + '/feature/test_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict), 0))

    enc = OneHotEncoder()
    for feature in cate_feature: # 25302
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))), 'csr', 'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=200)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict[feature].astype(str))), 'csr', 'bool')
    print('cv prepared !')

    sparse.save_npz(path + '/feature/test_train_csr.npz', base_train_csr)
    sparse.save_npz(path + '/feature/test_predict_csr.npz', base_predict_csr)

train_csr = sparse.hstack((num_train_csr, base_train_csr), 'csr', 'float32')
predict_csr = sparse.hstack((num_predict_csr, base_predict_csr), 'csr', 'float32')

feature_select = SelectPercentile(chi2, percentile=95)
feature_select.fit(train_csr, train_y)
train_csr = feature_select.transform(train_csr)
predict_csr = feature_select.transform(predict_csr)
print('feature select')
print(train_csr.shape)

lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=48, reg_alpha=3, reg_lambda=5,
    max_depth=-1, n_estimators=100, objective='binary', max_bin=150,
    subsample=0.75, colsample_bytree=0.8, subsample_freq=1,
    learning_rate=0.05, random_state=2018, n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=100)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
print(np.mean(best_score))
predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)

ll = log_loss(y_label, predict_result['predicted_score'].values)
print('logloss', ll)
