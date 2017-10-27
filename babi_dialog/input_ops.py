import numpy as np
import os
import pickle

from util import log

_TYPE_NUM = None


def parse_config_txt(path_to_config):
    config = dict()
    with open(os.path.join('babi_dialog', path_to_config), 'r') as f:
        text = f.readline().split('\t')
        config['memory_size'] = int(text[0])
        config['sentence_size'] = int(text[1])
        config['cand_sentence_size'] = int(text[2])
        config['cand_size'] = int(text[3])
        config['vocab_size'] = int(text[4])
        config['path_to_processed'] = text[5]
        config['_TYPE_NUM'] = int(text[6])
        global _TYPE_NUM
        _TYPE_NUM = int(text[6])
    return config


def load_data(task, type):
    config_txt = parse_config_txt('config_{}.txt'.format(task))
    if type == 'train':
        with open(os.path.join(config_txt['path_to_processed'], 'train_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    elif type == 'val':
        with open(os.path.join(config_txt['path_to_processed'], 'val_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    elif type == 'test':
        with open(os.path.join(config_txt['path_to_processed'], 'test_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    elif type == 'candidate':
        with open(os.path.join(config_txt['path_to_processed'], 'cand_set.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    elif type == 'idx_to_cand':
        with open(os.path.join(config_txt['path_to_processed'], 'idx_to_cand.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    elif type == 'idx_to_word':
        with open(os.path.join(config_txt['path_to_processed'], 'idx_to_word.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    return dataset


def batch_iter(c, c_real_len, q, q_real_len, cand, match_words, a_idx, shuffle, batch_size, num_epochs):
    c = np.array(c)
    c_real_len = np.array(c_real_len)
    q = np.array(q)
    q_real_len = np.array(q_real_len)
    cand_batch = np.tile(cand, (batch_size, 1, 1))
    match_words = np.array(match_words)
    a_idx = np.array(a_idx)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        log.infov("In epoch >> {}".format(epoch + 1))
        log.info("num batches per epoch is : {}".format(num_batches_per_epoch))
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            c_shuffled = c[shuffle_indices]
            c_real_len_shuffled = c_real_len[shuffle_indices]
            q_shuffled = q[shuffle_indices]
            q_real_len_shuffled = q_real_len[shuffle_indices]
            match_words_shuffled = match_words[shuffle_indices]
            a_idx_shuffled = a_idx[shuffle_indices]
        else:
            c_shuffled = c
            c_real_len_shuffled = c_real_len
            q_shuffled = q
            q_real_len_shuffled = q_real_len
            match_words_shuffled = match_words
            a_idx_shuffled = a_idx

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch = c_shuffled[start_index:end_index]
                c_real_len_batch = c_real_len_shuffled[start_index:end_index]
                q_batch = q_shuffled[start_index:end_index]
                q_real_len_batch = q_real_len_shuffled[start_index:end_index]
                match_words_batch = match_words_shuffled[start_index:end_index]
                cand_match_batch = []
                for match_word in match_words_batch:
                    cand_match = []
                    for candidate in cand:
                        match = [0] * _TYPE_NUM
                        if match_word[0]:
                            if match_word[0][-1] in candidate:
                                match[0] = 1
                        if match_word[1]:
                            if match_word[1][-1] in candidate:
                                match[1] = 1
                        if match_word[2]:
                            if match_word[2][-1] in candidate:
                                match[2] = 1
                        if match_word[3]:
                            if match_word[3][-1] in candidate:
                                match[3] = 1
                        if match_word[4]:
                            if match_word[4][-1] in candidate:
                                match[4] = 1
                        if match_word[5]:
                            if match_word[5][-1] in candidate:
                                match[5] = 1
                        if match_word[6]:
                            if match_word[6][-1] in candidate:
                                match[6] = 1
                        cand_match.append(match)
                    cand_match_batch.append(np.array(cand_match))
                cand_match_batch = np.array(cand_match_batch)
                a_idx_batch = a_idx_shuffled[start_index:end_index]
                yield list(zip(c_batch, c_real_len_batch, q_batch, q_real_len_batch, cand_batch, cand_match_batch,
                               a_idx_batch))


def batch_iter_test(c, c_real_len, q, q_real_len, cand, match_words, a_idx, batch_size):
    c = np.array(c)
    c_real_len = np.array(c_real_len)
    q = np.array(q)
    q_real_len = np.array(q_real_len)
    cand_batch = np.tile(cand, (batch_size, 1, 1))
    match_words = np.array(match_words)
    a_idx = np.array(a_idx)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        if end_index < data_size:
            c_batch = c[start_index:end_index]
            c_real_len_batch = c_real_len[start_index:end_index]
            q_batch = q[start_index:end_index]
            q_real_len_batch = q_real_len[start_index:end_index]
            match_words_batch = match_words[start_index:end_index]
            cand_match_batch = []
            for match_word in match_words_batch:
                cand_match = []
                for candidate in cand:
                    match = [0] * _TYPE_NUM
                    if match_word[0]:
                        if match_word[0][-1] in candidate:
                            match[0] = 1
                    if match_word[1]:
                        if match_word[1][-1] in candidate:
                            match[1] = 1
                    if match_word[2]:
                        if match_word[2][-1] in candidate:
                            match[2] = 1
                    if match_word[3]:
                        if match_word[3][-1] in candidate:
                            match[3] = 1
                    if match_word[4]:
                        if match_word[4][-1] in candidate:
                            match[4] = 1
                    if match_word[5]:
                        if match_word[5][-1] in candidate:
                            match[5] = 1
                    if match_word[6]:
                        if match_word[6][-1] in candidate:
                            match[6] = 1
                    cand_match.append(match)
                cand_match_batch.append(np.array(cand_match))
            cand_match_batch = np.array(cand_match_batch)
            a_idx_batch = a_idx[start_index:end_index]
        else:
            start_index = data_size - batch_size
            c_batch = c[start_index:]
            c_real_len_batch = c_real_len[start_index:]
            q_batch = q[start_index:]
            q_real_len_batch = q_real_len[start_index:]
            match_words_batch = match_words[start_index:]
            cand_match_batch = []
            for match_word in match_words_batch:
                cand_match = []
                for candidate in cand:
                    match = [0] * _TYPE_NUM
                    if match_word[0]:
                        if match_word[0][-1] in candidate:
                            match[0] = 1
                    if match_word[1]:
                        if match_word[1][-1] in candidate:
                            match[1] = 1
                    if match_word[2]:
                        if match_word[2][-1] in candidate:
                            match[2] = 1
                    if match_word[3]:
                        if match_word[3][-1] in candidate:
                            match[3] = 1
                    if match_word[4]:
                        if match_word[4][-1] in candidate:
                            match[4] = 1
                    if match_word[5]:
                        if match_word[5][-1] in candidate:
                            match[5] = 1
                    if match_word[6]:
                        if match_word[6][-1] in candidate:
                            match[6] = 1
                    cand_match.append(match)
                cand_match_batch.append(np.array(cand_match))
            cand_match_batch = np.array(cand_match_batch)
            a_idx_batch = a_idx[start_index:]
        yield list(zip(c_batch, c_real_len_batch, q_batch, q_real_len_batch, cand_batch, cand_match_batch, a_idx_batch))
