# -*- encoding: utf-8 -*-
'''
@File        :GBDT_LR.py
@Time        :2021/03/18 11:24:41
@Author      :Reid
@Version     :1.0
@Desc        :Facebook 2014 method GBDT + LR
'''

import lightgbm as lgb

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


print('load data...')
path = r'C:\Users\user\Downloads\Compressed\porto-seguro-safe-driver-prediction'

train = pd.read_csv(f'{path}/train.csv', sep=',', encoding='utf-8')
test = pd.read_csv(f'{path}/test.csv', sep=',', encoding='utf-8')

print('training shape:', train.shape)
print(train.columns)
print(test.columns)

numeric_cols = [
     "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",]

print(test.head(10))

X = train[numeric_cols]
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


num_leaf = 64

print('start training...')

gbm = lgb.train(
    params, lgb_train, valid_sets=lgb_train
)

print('Svae model...')

gbm.save_model('model.txt')

print('Start predicting...')

y_pred = gbm.predict(X_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:10])

print('Writing transformed training data')
transformed_training_matrix = np.zeros([
    len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)   # N * num_trees * num_leafs

for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1


y_pred_t  = gbm.predict(X_test, pred_leaf=True)

print('Writing transformed testing data')

transformed_testing_matrix = np.zeros([len(y_pred_t), len(y_pred_t[0]) * num_leaf], dtype=np.int64)

for i in range(0, len(y_pred_t)):
    temp = np.arange(len(y_pred_t[0])) * num_leaf + np.array(y_pred_t[i])
    transformed_testing_matrix[i][temp] += 1

lr = LogisticRegression(penalty='l2', C=0.05)
lr.fit(transformed_training_matrix, y_train)

y_pred_test = lr.predict_proba(transformed_testing_matrix)

print(y_pred_test)

NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
print("Normalized Cross Entropy " + str(NE))