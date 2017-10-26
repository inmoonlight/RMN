# based on ideas from https://github.com/voicy-ai/DialogStateTracking/blob/master/data/data_utils.py

import os
import pickle as pkl

import data_utils

DATA_DIR = './data/babi_dialog/'
P_DATA_DIR = './data/babi_dialog_pkl/'
if not os.path.exists(P_DATA_DIR):
    os.makedirs(P_DATA_DIR)


def prepare_data(task_id, is_oov=False):
    task_id = task_id
    is_oov = is_oov
    # get candidates (restaurants)
    candidates, candid2idx, idx2candid = data_utils.load_candidates(task_id=task_id,
                                                                    candidates_f=DATA_DIR + 'dialog-babi-candidates.txt')
    # get data
    train, test, val = data_utils.load_dialog_task(
        data_dir=DATA_DIR,
        task_id=task_id,
        candid_dic=candid2idx,
        isOOV=is_oov)
    ##
    # get metadata
    metadata = data_utils.build_vocab(train + test + val, candidates)

    ###
    # write data to file
    data_ = {
        'candidates': candidates,
        'train': train,
        'test': test,
        'val': val
    }
    if is_oov:
        with open(P_DATA_DIR + str(task_id) + '_oov.data.pkl', 'wb') as f:
            pkl.dump(data_, f)
    else:
        with open(P_DATA_DIR + str(task_id) + '.data.pkl', 'wb') as f:
            pkl.dump(data_, f)

    ### 
    # save metadata to disk
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid

    if is_oov:
        with open(P_DATA_DIR + str(task_id) + '_oov.metadata.pkl', 'wb') as f:
            pkl.dump(metadata, f)
    else:
        with open(P_DATA_DIR + str(task_id) + '.metadata.pkl', 'wb') as f:
            pkl.dump(metadata, f)
