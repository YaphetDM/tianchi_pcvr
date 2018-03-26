# coding:utf-8

from sklearn.metrics import log_loss
from utils import read_input
from .models.deep_cross_network import DeepCrossNetwork
from .models.pnn_keras import ProductNetwork

if __name__ == '__main__':
    train_file_path = '../data/train.txt'
    test_file_path = 'data/test'
    output_file_path = 'output.txt'
    is_train = True
    drop_pct = 0.98
    if is_train:
        featmap, train_real_value, train_discrete, train_labels, \
        valid_real_value, valid_discrete, valid_labels = read_input(train_file_path, test_file_path=None,
                                                                    is_train=is_train, drop_pct=drop_pct)
        features_len = len(featmap)
        print('features length: ', features_len)
        dcn = DeepCrossNetwork([4, 20], features_len, 4, 1024, 10, [128, 128, 128, 128, 128],
                               6e-4, 1e-4, 20, 0.01, 1024)
        opnn = ProductNetwork(20, features_len, 16, 150, [128, 128, 128, 128, 1], 50, 1024, 1e-5, 8e-4, 0.5, 0.01)
        dcn.train_with_valid([train_real_value, train_discrete], train_labels,
                             [valid_real_value, valid_discrete], valid_labels)
        opnn.train_with_valid(train_discrete, train_labels, valid_discrete, valid_labels)
        dcn_pred = dcn.dcn_predict([valid_real_value, valid_discrete])
        pnn_pred = opnn.pnn_predict(valid_discrete)
        print('log loss: ', log_loss(valid_labels, 0.5 * (dcn_pred + pnn_pred)))
        # dcn.train_with_valid([train_real_value, train_discrete], train_labels,
        #                      [train_real_value, train_discrete], train_labels)
    else:
        featmap, train_real_value, train_discrete, train_labels, \
        test_real_value, test_discrete, test_instance_id = read_input(train_file_path, test_file_path=test_file_path,
                                                                      is_train=False, drop_pct=drop_pct)
        features_len = len(featmap)
        print('features length: ', features_len)
        dcn = DeepCrossNetwork([4, 20], features_len, 4, 1024, 10, [128, 128, 128, 128, 128],
                               6e-4, 5e-5, 20, 0.01, 1024)
        opnn = ProductNetwork(20, features_len, 16, 150, [128, 128, 128, 128, 1], 50, 1024, 1e-5, 8e-4, 0.5, 0.01)
        dcn.train([train_real_value, train_discrete], train_labels)
        opnn.train(train_discrete, train_labels)
        dcn_pred = dcn.dcn_predict([test_real_value, test_discrete])
        pnn_pred = opnn.pnn_predict(test_discrete)
        predictions = 0.5*(dcn_pred+pnn_pred)
        with open(output_file_path, 'w') as f:
            f.write('instance_id predicted_score\n')
            for id, score in zip(test_instance_id, predictions):
                f.write(str(id) + ' ' + str(score) + '\n')
