# coding:utf-8

import time
from datetime import datetime
from keras.layers import Layer
from keras import backend as K
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


def get_day_hour(value, time_format='%Y-%m-%d-%H'):
    value = time.localtime(value)
    format_time = time.strftime(time_format, value)
    year, month, day, hour = format_time.split('-')
    dt = datetime(year=int(year), month=int(month), day=int(day))
    return int(day), str(dt.weekday()), hour


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


def long_tail(series, size, pct=0.95):
    idx = 0
    cnt = 0
    threshold = int(size * pct)
    feature_list = series.index.tolist()
    while cnt < threshold:
        cnt += series[feature_list[idx]]
        idx += 1
    return feature_list[:idx]


def extract_category_property(df):
    predict_category = df['predict_category_property'].map(
        lambda x: [v.split(':')[0] for v in x.split(';')])
    category_list = df['item_category_list'].map(lambda x: x.split(';'))
    train_category_join = category_list.combine(predict_category, lambda x, y: merge(x, y)).map(
        lambda x: ['category_join_' + str(s) for s in x])
    train_category_join_first = train_category_join.map(lambda x: x[0])
    train_category_join_second = train_category_join.map(lambda x: x[1])
    train_category_join_third = train_category_join.map(lambda x: x[2])
    df['category_join_first'] = train_category_join_first
    df['category_join_second'] = train_category_join_second
    df['category_join_third'] = train_category_join_third
    df.drop(['predict_category_property', 'item_category_list', 'item_property_list'], axis=1, inplace=True)


def extract_user_query(data):
    data['time'] = data['context_timestamp'].apply(get_day_hour)

    data['day'] = data.time.apply(lambda x: x[0])
    data['week'] = data.time.apply(lambda x: x[1])
    data['hour'] = data.time.apply(lambda x: x[2])
    # user_query_day = data.groupby(['user_id', 'day']).size(
    # ).reset_index().rename(columns={0: 'user_query_day'})
    # data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    user_query_week = data.groupby(['user_id', 'week']).size().reset_index().rename(
        columns={0: 'user_query_week'})
    data = pd.merge(data, user_query_week, 'left',
                    on=['user_id', 'week'])
    data.drop(['context_timestamp', 'user_id', 'time'], axis=1, inplace=True)
    return data


def read_input(train_file_path, test_file_path, is_train=True, drop_pct=0.95):
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

    raw_train_data = pd.read_table(train_file_path, sep=' ')

    # 去掉完全一样的数据
    raw_train_data.drop_duplicates(inplace=True)
    # 去掉无用特征

    raw_train_data.drop(useless_cols, axis=1, inplace=True)

    # 生成离散特征
    for col in discrete_cols:
        raw_train_data[col] = raw_train_data[col].map(lambda x: col + '_' + str(x))
    # 获取predict_category与category_list的交集, 将category_list长度扩展到3, 不够用0补齐
    extract_category_property(raw_train_data)
    # raw_train_data = extract_user_query(raw_train_data)
    raw_train_data['time'] = raw_train_data['context_timestamp'].apply(get_day_hour)
    raw_train_data['day'] = raw_train_data.time.apply(lambda x: x[0])
    raw_train_data['week'] = raw_train_data.time.apply(lambda x: 'week_' + str(x[1]))
    raw_train_data['hour'] = raw_train_data.time.apply(lambda x: 'hour_' + str(x[2]))
    if is_train:
        train = raw_train_data.loc[raw_train_data.day < 24]
        validate = raw_train_data.loc[raw_train_data.day == 24]
        # 对于实数特征缺失值补均值
        for k in real_value_cols:
            mean = train[k].mean()
            train[k].replace(-1, mean, inplace=True)
            validate[k].replace(-1, mean, inplace=True)

        # 特征统计
        features = []
        train_size = train.shape[0]
        for col in discrete_cols + create_cols:
            series = train[col].value_counts()
            if col in drop_long_tail_cols:
                features.extend(long_tail(series, size=train_size, pct=drop_pct))
            else:
                features.extend(series.index.tolist())

        features = [v for v in features if '_-1' not in v]
        # 生成featmap
        featmap = dict(zip(np.unique(features), range(1, len(features) + 1)))

        train_real_value = train[real_value_cols].applymap(lambda x: 0.1 * x)
        # 训练数据feature to index mapping
        train_discrete = train[discrete_cols + create_cols].applymap(lambda x: featmap.get(x, 0))
        train_labels = train['is_trade']

        validate_real_value = validate[real_value_cols].applymap(lambda x: 0.1 * x)
        # 测试数据feature to index mapping
        validate_discrete = validate[discrete_cols + create_cols].applymap(lambda x: featmap.get(x, 0))
        validate_labels = validate['is_trade']
        return featmap, train_real_value.values, train_discrete.values, train_labels.values, \
               validate_real_value.values, validate_discrete.values, validate_labels.values
        # return train_discrete
    else:
        raw_test_data = pd.read_table(test_file_path, sep=' ')
        extract_category_property(raw_test_data)
        # raw_test_data = extract_user_query(raw_test_data)
        raw_test_data['time'] = raw_test_data['context_timestamp'].apply(get_day_hour)
        raw_test_data['week'] = raw_test_data.time.apply(lambda x: 'week_' + str(x[1]))
        raw_test_data['hour'] = raw_test_data.time.apply(lambda x: 'hour_' + str(x[2]))
        # 生成离散特征
        for col in discrete_cols:
            raw_test_data[col] = raw_test_data[col].map(lambda x: col + '_' + str(x))

        for k in real_value_cols:
            mean = raw_train_data[k].mean()
            raw_train_data[k].replace(-1, mean, inplace=True)
            raw_test_data[k].replace(-1, mean, inplace=True)

        features = []
        train_size = raw_train_data.shape[0]
        for col in discrete_cols + create_cols:
            series = raw_train_data[col].value_counts()
            if col in drop_long_tail_cols:
                features.extend(long_tail(series, size=train_size, pct=drop_pct))
            else:
                features.extend(series.index.tolist())

        features = [v for v in features if '_-1' not in v]
        # 生成featmap
        featmap = dict(zip(np.unique(features), range(1, len(np.unique(features)) + 1)))

        train_real_value = raw_train_data[real_value_cols].applymap(lambda x: 0.1 * x)
        # 训练数据feature to index mapping
        train_discrete = raw_train_data[discrete_cols + create_cols].applymap(lambda x: featmap.get(x, 0))
        train_labels = raw_train_data['is_trade']

        test_real_value = raw_test_data[real_value_cols].applymap(lambda x: 0.1 * x)
        test_discrete = raw_test_data[discrete_cols + create_cols].applymap(lambda x: featmap.get(x, 0))
        test_instance_id = raw_test_data['instance_id']
        return featmap, train_real_value.values, train_discrete.values, train_labels.values, \
               test_real_value.values, test_discrete.values, test_instance_id.values


class _Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(_Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self._fix_unknown_dimension(
            input_shape[1:], self.target_shape)

    def call(self, inputs, **kwargs):
        # In case the target shape is not fully defined,
        # we need access to the shape of `inputs`.
        # solution: rely on `K.int_shape`.
        target_shape = self.target_shape
        if -1 in target_shape:
            input_shape = None
            try:
                input_shape = K.int_shape(inputs)
            except TypeError:
                pass
            if input_shape is not None:
                target_shape = self.compute_output_shape(input_shape)[1:]
        return K.reshape(inputs, (-1,) + target_shape)

    def compute_mask(self, inputs, mask=None):
        return mask


if __name__ == '__main__':
    train_file_path = 'data/train.txt'
    test_file_path = 'data/train.txt'
    featmap, train_real_value, train_discrete, train_labels, \
    test_real_value, test_discrete, test_instance_id = read_input(train_file_path, test_file_path)
    for each in featmap:
        print(each, featmap[each])
        print('test')
