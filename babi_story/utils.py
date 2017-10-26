import numpy as np
import os
import pickle
import sys

sys.path.append('../../')

import alarm


def read_data_test(file_path):
    with open(os.path.join(file_path, 'test_dataset.pkl'), 'rb') as f:
        test = pickle.load(f)
    [test_q, test_a, test_a_num, test_c, test_l, test_c_real_len, test_q_real_len, test_hint] = test
    with open(os.path.join(file_path, 'answer_word_dict.pkl'), 'rb') as f:
        word_set = pickle.load(f)
    return [test_q, test_a, test_a_num, test_c, test_l, test_c_real_len, test_q_real_len, test_hint], word_set


def read_data(file_path):
    with open(os.path.join(file_path, 'train_dataset.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(file_path, 'val_dataset.pkl'), 'rb') as f:
        val = pickle.load(f)
    [train_q, train_a, train_a_num, train_c, train_label, train_c_real_len, train_q_real_len, train_hint] = train
    [val_q, val_a, val_a_num, val_c, val_label, val_c_real_len, val_q_real_len, val_hint] = val

    return ([train_q, train_a, train_a_num, train_c, train_label, train_c_real_len, train_q_real_len, train_hint], \
            [val_q, val_a, val_a_num, val_c, val_label, val_c_real_len, val_q_real_len, val_hint])


def batch_iter(c, q, l, a, a_num, c_real_len, q_real_len, h, batch_size, num_epochs, _version, c_max_len=130, shuffle=True,
               is_training=True):
    """
    Generates a batch iterator for a dataset.
    """
    c = np.array(c)
    q = np.array(q)
    l = np.array(l)
    a = np.array(a)
    a_num = np.array(a_num)
    c_real_len = np.array(c_real_len)
    q_real_len = np.array(q_real_len)
    h = np.array(h)
    label_matrix = np.append(np.eye(c_max_len), [[0] * c_max_len], axis=0)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        try:
            if is_training:
                alarm.send_message('Model-v{} training...epoch {}'.format(_version, epoch + 1), channel='babi')
        except:
            pass
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            c_shuffled = c[shuffle_indices]
            q_shuffled = q[shuffle_indices]
            l_shuffled = l[shuffle_indices]
            a_shuffled = a[shuffle_indices]
            a_num_shuffled = a_num[shuffle_indices]
            c_real_len_shuffled = c_real_len[shuffle_indices]
            q_real_len_shuffled = q_real_len[shuffle_indices]
            h_shuffled = h[shuffle_indices]
        else:
            c_shuffled = c
            q_shuffled = q
            l_shuffled = l
            a_shuffled = a
            a_num_shuffled = a_num
            c_real_len_shuffled = c_real_len
            q_real_len_shuffled = q_real_len
            h_shuffled = h

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch, h_batch = c_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               q_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               l_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               a_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               a_num_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               c_real_len_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               q_real_len_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               h_shuffled[
                                                                                                               start_index:end_index]
                h_batch_result = []
                for i in range(batch_size):
                    h_batch_result.append(np.sum(np.eye(c_max_len)[h_batch[i]], axis=0))
                yield list(
                    zip(c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch,
                        q_real_len_batch, np.array(h_batch_result)))


def batch_iter_test(c, q, l, a, a_num, c_real_len, q_real_len, h, batch_size, _version, c_max_len, num_epochs=1, shuffle=False):
    """
    Generates a batch iterator for a test dataset.
    """
    c = np.array(c)
    q = np.array(q)
    l = np.array(l)
    a = np.array(a)
    a_num = np.array(a_num)
    c_real_len = np.array(c_real_len)
    q_real_len = np.array(q_real_len)
    h = np.array(h)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        print("Testing...")
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            c_shuffled = c[shuffle_indices]
            q_shuffled = q[shuffle_indices]
            l_shuffled = l[shuffle_indices]
            a_shuffled = a[shuffle_indices]
            a_num_shuffled = a_num[shuffle_indices]
            c_real_len_shuffled = c_real_len[shuffle_indices]
            q_real_len_shuffled = q_real_len[shuffle_indices]
            h_shuffled = h[shuffle_indices]
        else:
            c_shuffled = c
            q_shuffled = q
            l_shuffled = l
            a_shuffled = a
            a_num_shuffled = a_num
            c_real_len_shuffled = c_real_len
            q_real_len_shuffled = q_real_len
            h_shuffled = h

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch, h_batch = c_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               q_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               l_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               a_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               a_num_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               c_real_len_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               q_real_len_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               h_shuffled[
                                                                                                               start_index:end_index]
            else:
                end_index = data_size
                start_index = end_index - batch_size
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch, h_batch = c_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               q_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               l_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               a_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               a_num_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               c_real_len_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               q_real_len_shuffled[
                                                                                                               start_index:end_index], \
                                                                                                               h_shuffled[
                                                                                                               start_index:end_index]
            h_batch_result = []
            for i in range(batch_size):
                h_batch_result.append(np.sum(np.eye(c_max_len)[h_batch[i]], axis=0))
            yield list(zip(c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch,
                           np.array(h_batch_result)))


def parse_config(string):
    # parsing txt file
    result = dict()
    args = string.split("\t")
    result['c_max_len'] = int(args[0])
    result['s_max_len'] = int(args[1])
    result['q_max_len'] = int(args[2])
    result['babi_processed'] = args[3]  # path
    result['batch_size'] = int(args[4])
    result['s_hidden'] = int(args[5])
    result['learning_rate'] = float(args[6])
    result['iter_time'] = int(args[7])
    result['display_step'] = int(args[8])
    result['vocab_size'] = int(args[9])
    return result

def batch_iter_cl(c, q, l, a, a_num, c_real_len, q_real_len, h, batch_size, epoch,
                  curriculum_order, _version, update_period, shuffle=True,is_training=True):
    """
    Generates a batch iterator for a dataset.
    """
    c = np.array(c)
    q = np.array(q)
    l = np.array(l)
    a = np.array(a)
    a_num = np.array(a_num)
    c_real_len = np.array(c_real_len)
    q_real_len = np.array(q_real_len)
    h = np.array(h)
    to = np.array(curriculum_order)
    label_matrix = np.append(np.eye(130), [[0] * 130], axis=0)
    data_size = len(to)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    print("In epoch >> " + str(epoch + 1))
    print("num batches per epoch is: " + str(num_batches_per_epoch))
    try:
        if is_training:
            alarm.send_message('Model-v{} training...epoch {}'.format(_version, (epoch + 1) * update_period))
    except:
        pass

    # Shuffle the data at each epoch
    if shuffle:
        shuffled_indices = np.random.permutation(to)
    else:
        shuffled_indices = to
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        if end_index < data_size:
            c_batch = c[shuffled_indices[start_index:end_index]]
            q_batch = q[shuffled_indices[start_index:end_index]]
            l_batch = l[shuffled_indices[start_index:end_index]]
            a_batch = a[shuffled_indices[start_index:end_index]]
            a_num_batch = a_num[shuffled_indices[start_index:end_index]]
            c_real_len_batch = c_real_len[shuffled_indices[start_index:end_index]]
            q_real_len_batch = q_real_len[shuffled_indices[start_index:end_index]]
            h_batch = h[shuffled_indices[start_index:end_index]]
            h_batch_result = []
            for i in range(batch_size):
                h_batch_result.append(np.sum(np.eye(130)[h_batch[i]], axis=0))
            yield list(
                zip(c_batch, q_batch,l_batch, a_batch, a_num_batch, c_real_len_batch,
                    q_real_len_batch, np.array(h_batch_result), shuffled_indices[start_index:end_index]))            
        else:
            end_index = data_size
            start_index = end_index - batch_size
            c_batch = c[shuffled_indices[start_index:end_index]]
            q_batch = q[shuffled_indices[start_index:end_index]]
            l_batch = l[shuffled_indices[start_index:end_index]]
            a_batch = a[shuffled_indices[start_index:end_index]]
            a_num_batch = a_num[shuffled_indices[start_index:end_index]]
            c_real_len_batch = c_real_len[shuffled_indices[start_index:end_index]]
            q_real_len_batch = q_real_len[shuffled_indices[start_index:end_index]]
            h_batch = h[shuffled_indices[start_index:end_index]]
            h_batch_result = []
            for i in range(batch_size):
                h_batch_result.append(np.sum(np.eye(130)[h_batch[i]], axis=0))
            yield list(
                zip(c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch,
                    q_real_len_batch, np.array(h_batch_result), shuffled_indices[start_index:end_index]))                        

def make_curriculum_order(curriculum_dict, curriculum_count, update_period):
    value_sum = 0
    for key in curriculum_dict.keys():
        tmp_value = curriculum_dict[key] / curriculum_count[key]
        value_sum += tmp_value
        curriculum_dict[key] = tmp_value

    length = len(list(curriculum_dict.keys()))
    train_order = []
    for key in curriculum_dict.keys():
        repeat = int(curriculum_dict[key]/value_sum * length * (update_period-1))+1
        train_order.extend([key] * repeat)
    return train_order            

