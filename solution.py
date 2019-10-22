# -*- coding: UTF-8 -*-

from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import datetime
import time
import os

path = '.'

warnings.filterwarnings("ignore")


def get_ctr_matrix(data, src, dest):
    
    temp = data.loc[data['label'] != -1, [src, dest, 'label']]
    
    src_gb = temp.groupby([src])
    src_frame = src_gb.agg({'label': 'sum'}).rename(columns={'label': 'src_sum'}).join(
        src_gb.agg({'label': 'count'}).rename(columns={'label': 'src_count'})).reset_index()
    del src_gb

    dest_gb = temp.groupby([src, dest])
    dest_frame = dest_gb.agg({'label': 'sum'}).rename(columns={'label': 'dest_sum'}).join(
        dest_gb.agg({'label': 'count'}).rename(columns={'label': 'dest_count'})).reset_index()
    del dest_gb

    frame = pd.merge(src_frame, dest_frame, on=[src], how='right')
    del temp, src_frame, dest_frame

    frame['rate'] = (frame['src_sum'] + frame['dest_sum']) / (frame['src_count'] + frame['dest_count'])
    frame.drop(['src_sum', 'src_count', 'dest_sum', 'dest_count'], axis=1, inplace=True)
    data = pd.merge(data, frame, on=[src, dest], how='left').fillna(0)
    del frame

    src_hot = pd.get_dummies(data[src], dtype='float32')
    for h in src_hot.columns:
        src_hot[h] *= data['rate']
    data.drop(['rate'], axis=1, inplace=True)
    
    return sparse.csr_matrix(src_hot[data['label'] != -1], dtype='float32'), sparse.csr_matrix(src_hot[data['label'] == -1], dtype='float32')


train = pd.read_table(path + '/Data/round1_iflyad_train.txt')
test = pd.read_table(path + '/Data/round1_iflyad_test_feature.txt')
data = pd.concat([train, test], axis=0, ignore_index=True)

data = data.fillna(-1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)
del data['click']

###----------------- pretreatment -----------------###

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

###-------------- ctr pretreatment ----------------###
data.loc[data['make'] < 200, 'make_cate'] = data.loc[data['make'] < 200, 'make']
data['make_cate'] = data['make_cate'].fillna(200).astype('int32')

###----------------- ctr feature ------------------###
ctr_list = [
    # ad
    ('creative_width', 'app_id'),
    ('creative_width', 'inner_slot_id'),
    ('creative_height', 'inner_slot_id'),
    ('advert_id', 'app_cate_id'),
    ('advert_id', 'app_id'),
    ('advert_id', 'inner_slot_id'),
    ('advert_id', 'creative_id'),
    ('campaign_id', 'inner_slot_id'),
    ('campaign_id', 'app_id'),
    ('campaign_id', 'app_cate_id'),
    ('campaign_id', 'creative_id'),
    ('creative_tp_dnf', 'inner_slot_id'),
    ('creative_type', 'make_cate'),
    ('creative_tp_dnf', 'app_id'),
    # user
    ('nnt', 'creative_id'),
    ('nnt', 'app_cate_id'),
    # app
    ('app_cate_id', 'make_cate'),
    ('app_cate_id', 'creative_id'),
]

ctr_feature = []
for (src, dest) in ctr_list:
    ctr_feature += [src]

###------------------------------------------------###

cate_feature = [x for x in origin_cate_list if x not in ctr_feature]
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
for (src, dest) in ctr_list:
    ctr_t, ctr_p = get_ctr_matrix(data, src, dest)
    ctr_train_csr = sparse.hstack((ctr_train_csr, ctr_t), 'csr', 'float32')
    ctr_predict_csr = sparse.hstack((ctr_predict_csr, ctr_p), 'csr', 'float32')
    print(src, dest, 'over')
print('ctr feature prepared !')

num_train_csr = sparse.hstack((ctr_train_csr, sparse.csr_matrix(train_x[num_feature])), 'csr', 'float32')
num_predict_csr = sparse.hstack((ctr_predict_csr, sparse.csr_matrix(predict[num_feature])), 'csr', 'float32')
print('number feature prepared !')

# 默认加载 如果 增加了cate类别特征 请改成false重新生成
if os.path.exists(path + '/feature/base_train_csr.npz') and False:
    print('load_csr---------')
    base_train_csr = sparse.load_npz(path + '/feature/base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz(path + '/feature/base_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))), 'csr', 'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype('str')
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype('str'))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict[feature].astype('str'))), 'csr', 'bool')
    print('cv prepared !')

    sparse.save_npz(path + '/feature/base_train_csr.npz', base_train_csr)
    sparse.save_npz(path + '/feature/base_predict_csr.npz', base_predict_csr)

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
    max_depth=-1, n_estimators=2000, objective='binary', max_bin=150,
    subsample=0.75, colsample_bytree=0.8, subsample_freq=1,
    learning_rate=0.05, random_state=2018, n_jobs=-1
)

best_score = []
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
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

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv(path + "/Submit/lgb_baseline_%s.csv" % now, index=False)
