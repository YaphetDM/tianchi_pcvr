# coding:utf-8
from codecs import open


def parse(path=None):
    _len = 0
    feature = None
    feature_cnt = {}
    with open(path, encoding='utf8') as content:
        for line in content.readlines():
            if line.startswith('instance'):
                feature = line.strip().split(' ')
                _len = len(feature)
            else:
                split = line.strip().split(' ')
                for i in range(1, _len - 1):
                    value = split[i]
                    if ';' not in value:
                        feature_value = feature[i]+'_'+value
                        if feature_value not in feature_cnt:
                            feature_cnt.setdefault(feature_value,1)
                        else:
                            feature_cnt[feature_value] += 1
                    # if ';' in split[i]:
                    #     [feature[i] + '_' + v for v in split[i].split(';')]

    return feature_cnt


if __name__ == '__main__':
    path = 'data/train.txt'
    for v in parse(path):
        if 'score' in v:
            print(v)
    print(len(parse(path)))
