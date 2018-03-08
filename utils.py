# coding:utf-8
import time
from codecs import open
from datetime import datetime
import numpy as np


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


def add_dict(feat_value, _dict=None):
    if feat_value not in _dict:
        _dict.setdefault(feat_value, 1)
    else:
        _dict[feat_value] += 1


def get_week_day(value, _format='%Y-%m-%d-%H'):
    value = time.localtime(value)
    format_time = time.strftime(_format, value)
    year, month, day, hour = format_time.split('-')
    dt = datetime(year=int(year), month=int(month), day=int(day))
    return dt.weekday(), hour


def get_predict_category_property(_str=None):
    res = []
    for each in _str.split(';'):
        if each.split(':')[1] == '-1':
            continue
        else:
            category = each.split(':')[0]
            res.extend(['predict_category_property_' + category + '_' + pro
                        for pro in each.split(':')[1].split(',')])
    return res


# def yield_train_data()

def input_data(path=None, df=5):
    _len = 0
    feature = None
    feature_cnt = {}
    feature_all = []
    labels = []
    with open(path, encoding='utf8') as content:
        for line in content.readlines():
            if line.startswith('instance'):
                feature = line.strip().split(' ')
                _len = len(feature)
            else:
                labels.append(float(line.strip().split(' ')[-1]))
                feature_each = []
                scores = []
                split = line.strip().split(' ')
                for i in range(1, _len - 1):
                    # item_category_list -> 2, item_property_list -> 3
                    if i == 2 or i == 3:
                        feat = feature[i].replace('_list', '')
                        value = split[i]
                        tmp = [feat + '_' + v for v in value.split(';')]
                        for v in tmp:
                            feature_each.append(v)
                            add_dict(v, feature_cnt)
                    # context_timestamp
                    elif i == 16:
                        value = int(split[i])
                        week, hour = get_week_day(value)
                        feature_each.append('week_' + str(week))
                        feature_each.append('hour_' + hour)
                        add_dict('week_' + str(week), feature_cnt)
                        add_dict('hour_' + hour, feature_cnt)
                    # predict_category_property
                    # 5755694407684602296:2636395404473730413;8710739180200009128:-1;
                    # 7908382889764677758:2636395404473730413;9121432215720987772:-1;
                    # 8257512457089702259:-1;8896700187874717254:-1
                    elif i == 18:
                        # for each in get_predict_category_property(split[i]):
                        #     feature_each.append(each)
                        #     add_dict(each, feature_cnt)
                        pass

                    else:
                        value = split[i]
                        # user_id,context_id,shop_review_positive_rate,shop_score_service,shop_score_delivery,shop_score_description
                        if i not in [10, 15, 21, 23, 24, 25] and value != '-1':
                            feature_value = feature[i] + '_' + value
                            feature_each.append(feature_value)
                            add_dict(feature_value, feature_cnt)
                        # shop_review_positive_rate,shop_score_service,shop_score_delivery,shop_score_description
                        elif i in [21, 23, 24, 25]:
                            scores.append(float(split[i]))
                feature_each.extend(scores)
                feature_all.append(feature_each)
    filter_cnt = {key: feature_cnt[key] for key in feature_cnt.keys() if feature_cnt[key] >= df}

    featmap = dict(zip(sorted(filter_cnt.keys()), range(len(filter_cnt))))
    sample = np.array([each[-4:] + [featmap.get(v, -1) for v in each[:-4] if featmap.get(v, -1) >= 0]
                       for each in feature_all])
    sample_len = len(sample)
    perm = np.arange(sample_len)
    np.random.shuffle(perm)
    sample = sample[perm]

    # 7 2 1
    train_idx = int(sample_len * 0.7)
    valuate_idx = int(sample_len * 0.9)

    train_features = sample[:train_idx]
    train_labels = labels[:train_idx]

    valuate_features = sample[train_idx:valuate_idx]
    valuate_labels = labels[train_idx:valuate_idx]

    test_features = sample[valuate_idx:]
    test_labels = labels[valuate_idx:]

    train = DataSet(train_features, train_labels)
    valuate = DataSet(valuate_features, valuate_labels)
    test = DataSet(test_features, test_labels)

    return train, valuate, test, featmap
    # return feature_cnt, feature_all


if __name__ == '__main__':
    path = 'data/train.txt'
    train, valuate, test, featmap = input_data(path)
    for i in featmap:
        print(i, featmap[i])
    for each in train.features()[0:5]:
        print(len(each))
