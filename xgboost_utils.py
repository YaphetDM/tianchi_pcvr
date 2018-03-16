# coding:utf-8

import time
from datetime import datetime

import numpy as np
import pandas as pd


class DataSet(object):
    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = features.shape[0]

    def features(self):
        return self._features

    def labels(self):
        return self._labels

    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size=64):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end].tolist(), self._labels[start:end].tolist()


def get_day_hour(value, day_format='%Y-%m-%d', time_format='%Y-%m-%d-%H'):
    value = time.localtime(value)
    format_day = time.strftime(day_format, value)
    format_time = time.strftime(time_format, value)
    year, month, day, hour = format_time.split('-')
    dt = datetime(year=int(year), month=int(month), day=int(day))
    return format_day, str(dt.weekday()), hour


def merge(x, y):
    tmp = []
    for i in range(len(x)):
        if x[i] in y:
            tmp.append(x[i])
        else:
            tmp.append(0)
    if len(tmp) == 3:
        return tmp
    elif len(tmp) == 2:
        return tmp + [0]
    elif len(tmp) == 1:
        return tmp + [0, 0]
    else:
        return [0, 0, 0]


def long_tail(series, size, pct=0.99):
    idx = 0
    cnt = 0
    threshold = int(size * pct)
    feature_list = series.index.tolist()
    while cnt < threshold:
        cnt += series[feature_list[idx]]
        idx += 1
    return feature_list[:idx]


def read_input(file_path, cond_day='2018-09-23'):
    # 无用特征
    useless_cols = ['instance_id', 'user_id', 'context_id']

    # 离散特征
    discrete_cols = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                     'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level',
                     'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_id',
                     'shop_review_num_level', 'shop_star_level']

    # 需要去长尾的特征
    drop_long_tail_cols = ['item_id', 'item_brand_id', 'shop_id']

    # 实值特征
    real_value_cols = ['shop_review_positive_rate', 'shop_score_service',
                       'shop_score_delivery', 'shop_score_description']

    # 新建特征
    create_cols = ['hour', 'week', 'category_join_first', 'category_join_second', 'category_join_third']

    raw_data = pd.read_table(file_path, sep=' ')
    # 去掉完全一样的数据
    raw_data.drop_duplicates(inplace=True)
    # 去掉无用特征
    raw_data.drop(useless_cols, axis=1, inplace=True)

    # 生成离散特征
    for col in discrete_cols:
        raw_data[col] = raw_data[col].map(lambda x: col + '_' + str(x))

    # 获取predict_category与category_list的交集, 将category_list长度扩展到3, 不够用0补齐
    predict_category = raw_data['predict_category_property'].map(lambda x: [v.split(':')[0] for v in x.split(';')])
    category_list = raw_data['item_category_list'].map(lambda x: x.split(';'))
    category_join = category_list.combine(predict_category, lambda x, y: merge(x, y)).map(
        lambda x: ['category_join_' + str(s) for s in x])
    category_join_first = category_join.map(lambda x: x[0])
    category_join_second = category_join.map(lambda x: x[1])
    category_join_third = category_join.map(lambda x: x[2])
    raw_data['category_join_first'] = category_join_first
    raw_data['category_join_second'] = category_join_second
    raw_data['category_join_third'] = category_join_third
    raw_data.drop(['predict_category_property', 'item_category_list', 'item_property_list'], axis=1, inplace=True)

    # 将context_timestamp分解成day weekday hour
    maps = {'day': 0, 'week': 1, 'hour': 2}
    for j in maps:
        raw_data[j] = raw_data['context_timestamp'].map(lambda x: j + '_' + get_day_hour(x)[maps[j]])

    # 构建训练数据和测试数据
    train = raw_data.where(raw_data['day'] <= 'day_' + cond_day).dropna(axis=0)
    test = raw_data.where(raw_data['day'] > 'day_' + cond_day).dropna(axis=0)

    # 去除context_timestamp和day
    train.drop(['context_timestamp', 'day'], axis=1, inplace=True)
    test.drop(['context_timestamp', 'day'], axis=1, inplace=True)

    # 对于实数特征缺失值补均值
    for k in real_value_cols:
        mean = train[k].mean()
        train[k].replace(-1, mean, inplace=True)
        test[k].replace(-1, mean, inplace=True)

    # 特征统计
    features = []
    train_size = train.shape[0]
    for col in discrete_cols + create_cols:
        series = train[col].value_counts()
        if col in drop_long_tail_cols:
            features.extend(long_tail(series, size=train_size))
        else:
            features.extend(series.index.tolist())

    features = [v for v in features if '_-1' not in v]
    # 生成featmap
    featmap = dict(zip(np.unique(features), range(1, len(features) + 1)))

    train_real_value = train[real_value_cols].applymap(lambda x: 0.1*x)
    # 训练数据feature to index mapping
    train_discrete = train[discrete_cols + create_cols].applymap(lambda x: featmap.get(x, 0))
    train_labels = train['is_trade']

    test_real_value = test[real_value_cols].applymap(lambda x: 0.1*x)
    # 测试数据feature to index mapping
    test_discrete = test[discrete_cols + create_cols].applymap(lambda x: featmap.get(x, 0))
    test_labels = test['is_trade']

    return featmap, train_real_value.values, train_discrete.values, train_labels.values, \
        test_real_value.values, test_discrete.values, test_labels.values


if __name__ == '__main__':
    file_path = 'data/train.txt'
    featmap, train_real_value, train_discrete, train_labels, \
        test_real_value, test_discrete, test_labels = read_input(file_path)
    print(len(featmap))
