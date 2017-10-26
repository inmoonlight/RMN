# based on ideas from https://github.com/voicy-ai/DialogStateTracking/blob/master/data/data_utils.py

import re
import os

from itertools import chain
from six.moves import reduce

DATA_SOURCE = './data/babi_dialog/dialog-babi-candidates.txt'
STOP_WORDS = set(["a", "an", "the"])


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent = sent.lower()
    if sent == '<silence>':
        return [sent]
    result = [x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in STOP_WORDS]
    if not result:
        result = ['<silence>']
    if result[-1] == '.' or result[-1] == '?' or result[-1] == '!':
        result = result[:-1]
    return result


def load_candidates(task_id, candidates_f=DATA_SOURCE):
    # containers
    candidates, candid2idx, idx2candid = [], {}, {}
    # update data source file based on task id
    # read from file
    with open(candidates_f) as f:
        # iterate through lines
        for i, line in enumerate(f):
            # tokenize each line into... well.. tokens!
            candid2idx[line.strip().split(' ', 1)[1]] = i
            candidates.append(tokenize(line.strip()))
            idx2candid[i] = line.strip().split(' ', 1)[1]
    return candidates, candid2idx, idx2candid


def parse_dialogs_per_response(lines, candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data = []
    context = []
    u = None
    r = None
    for line in lines:
        line = line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = line.split('\t')
                a = candid_dic[r]
                u = tokenize(u)
                r = tokenize(r)
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))
                data.append((context[:], u[:], a))
                u.append('$u')
                u.append('#' + str(nid))
                r.append('$r')
                r.append('#' + str(nid))
                context.append(u)
                context.append(r)
            else:
                r = tokenize(line)
                r.append('$r')
                r.append('#' + str(nid))
                context.append(r)
        else:
            # clear context
            context = []
    return data


def get_dialogs(f, candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f, encoding='utf8') as f:
        return parse_dialogs_per_response(f.readlines(), candid_dic)


def load_dialog_task(data_dir, task_id, candid_dic, isOOV=False):
    '''Load the nth task. 
    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 6

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else:
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file, candid_dic)
    test_data = get_dialogs(test_file, candid_dic)
    val_data = get_dialogs(val_file, candid_dic)
    return train_data, test_data, val_data


def build_vocab(data, candidates):
    """
    data: train + val + test
    """
    vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a in data))
    vocab |= reduce(lambda x, y: x | y, (set(candidate) for candidate in candidates))
    vocab = sorted(vocab)
    w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (s for s, _, _ in data)))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    candidate_sentence_size = max(map(len, candidates))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = max_story_size
    vocab_size = len(w2idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position

    return {
        'w2idx': w2idx,
        'idx2w': vocab,
        'sentence_size': sentence_size,
        'candidate_sentence_size': candidate_sentence_size,
        'memory_size': memory_size,
        'vocab_size': vocab_size,
        'n_cand': len(candidates)
    }  # metadata
