import numpy as np
import os
import pickle


def parse_config(string):
    # parsing txt file
    result = dict()
    args = string.split("\t")
    result['c_max_len'] = int(args[0])
    result['s_max_len'] = int(args[1])
    result['q_max_len'] = int(args[2])
    result['babi_processed'] = args[3]
    result['vocab_size'] = int(args[4])
    return result


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


def batch_iter(c, q, l, a, a_num, c_real_len, q_real_len, batch_size, num_epochs, shuffle=True):
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
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
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
        else:
            c_shuffled = c
            q_shuffled = q
            l_shuffled = l
            a_shuffled = a
            a_num_shuffled = a_num
            c_real_len_shuffled = c_real_len
            q_real_len_shuffled = q_real_len

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, \
                c_real_len_batch, q_real_len_batch = c_shuffled[start_index:end_index], \
                                                     q_shuffled[start_index:end_index], \
                                                     l_shuffled[start_index:end_index], \
                                                     a_shuffled[start_index:end_index], \
                                                     a_num_shuffled[start_index:end_index], \
                                                     c_real_len_shuffled[start_index:end_index], \
                                                     q_real_len_shuffled[start_index:end_index]
                yield list(zip(c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch))


def batch_iter_test(c, q, l, a, a_num, c_real_len, q_real_len, batch_size):
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
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    print("Testing...")
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        if end_index < data_size:
            c_batch, q_batch, l_batch, a_batch, a_num_batch, \
            c_real_len_batch, q_real_len_batch = c[start_index:end_index], \
                                                 q[start_index:end_index], \
                                                 l[start_index:end_index], \
                                                 a[start_index:end_index], \
                                                 a_num[start_index:end_index], \
                                                 c_real_len[start_index:end_index], \
                                                 q_real_len[start_index:end_index]
        else:
            end_index = data_size
            start_index = end_index - batch_size
            c_batch, q_batch, l_batch, a_batch, a_num_batch, \
            c_real_len_batch, q_real_len_batch = c[start_index:end_index], \
                                                 q[start_index:end_index], \
                                                 l[start_index:end_index], \
                                                 a[start_index:end_index], \
                                                 a_num[start_index:end_index], \
                                                 c_real_len[start_index:end_index], \
                                                 q_real_len[start_index:end_index]
        yield list(zip(c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch))
