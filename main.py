import numpy as np
import os
from data_preprocess import read_data
from model import MemN2N
import tensorflow as tf

def main():

    config = {
        'batch_size'    : 128,
        'emb_dim'       : 150,
        'mem_size'      : 100,
        'test'          : False,
        'n_epoch'       : 50,
        'n_hop'         : 6,
        'n_words'       : None,
        'lr'            : 0.001,
        'std_dev'       : 0.05,
        'cp_dir'        : 'checkpoints'

    }

    count = list()
    word2idx = dict()
    train_data = read_data('./data/ptb.train.txt', count, word2idx)
    valid_data = read_data('./data/ptb.valid.txt', count, word2idx)
    test_data = read_data('./data/ptb.test.txt', count, word2idx)

    config['n_words'] = len(word2idx)
    
    with tf.Session() as sess:
        print "Training..."
        mod = MemN2N(config, sess)
        mod.train(train_data,valid_data)
        mod.test(test_data)


if __name__ == '__main__':
    main()