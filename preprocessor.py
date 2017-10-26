import argparse
import numpy as np
import pickle
import os
import re

from tqdm import tqdm

from util import log

## match type for babi dialog task
_CUISINE = ['british', 'cantonese', 'french', 'indian', 'italian', 'japanese', 'korean', 'spanish', 'thai',
            'vietnamese']
_LOCATION = ['bangkok', 'beijing', 'bombay', 'hanoi', 'london', 'madrid', 'paris', 'rome', 'seoul', 'tokyo']
_PRICE = ['cheap', 'moderate', 'expensive']
_RATING = ['1', '2', '3', '4', '5', '6', '7', '8']
_PHONE = ['R_phone']
_ADDRESS = ['R_address']
_NUMBER = ['two', 'four', 'six', 'eight']
_TYPE_NUM = 7


class Preprocess:
    def __init__(self, config):
        if config.data == 'story':
            self.path_to_babi = './data/babi_story'
            self.train_paths = None
            self.val_paths = None
            self.test_paths = None
            self.path_to_processed = './babi_story/story_processed'
            self._cqa_word_set = set()
            self.c_max_len = 0
            self.s_max_len = 0
            self.q_max_len = 0
            self.word_count = 0
            self.mask_index = 0

            self.run_story()

        elif config.data == 'dialog':
            from data_to_pickle import prepare_data
            self.path = './data/babi_dialog_pkl'

            for task_id in range(1, 6):
                for is_oov in [False, True]:
                    prepare_data(task_id=task_id, is_oov=is_oov)
                    self.task = str(task_id)
                    self.is_oov = is_oov

                    if self.is_oov:
                        filename_data, filename_meta = self.task + "_oov.data.pkl", self.task + "_oov.metadata.pkl"
                        self.path_to_processed = './babi_dialog/dialog_oov_processed_task_' + str(self.task)
                    else:
                        filename_data, filename_meta = self.task + ".data.pkl", self.task + ".metadata.pkl"
                        self.path_to_processed = './babi_dialog/dialog_processed_task_' + str(self.task)

                    if not os.path.exists(self.path_to_processed):
                        os.makedirs(self.path_to_processed)

                    with open(os.path.join(self.path, filename_data), 'rb') as f:
                        data_ = pickle.load(f)
                    with open(os.path.join(self.path, filename_meta), 'rb') as f:
                        metadata = pickle.load(f)

                    self.candidates = data_['candidates']
                    self.candid2idx, self.idx2candid = metadata['candid2idx'], metadata['idx2candid']

                    # get train/test/val data
                    self.train, self.test, self.val = data_['train'], data_['test'], data_['val']

                    # gather more information from metadata
                    self.sentence_size = metadata['sentence_size']
                    self.w2idx = metadata['w2idx']
                    self.idx2w = metadata['idx2w']
                    self.memory_size = max(map(len, (s for s, _, _ in self.train + self.val + self.test)))
                    self.vocab_size = metadata['vocab_size']
                    self.n_cand = metadata['n_cand']
                    self.candidate_sentence_size = metadata['candidate_sentence_size']

                    self.run_dialog()

    ### story preprocessing ###

    def run_story(self):
        self.set_path()
        self._set_word_set()
        self.load_train()
        self.load_val()
        self.load_test()
        ### write data info to config.txt
        with open(os.path.join('./babi_story', 'config.txt'), 'w') as f:
            f.write(str(self.c_max_len) + "\t")
            f.write(str(self.s_max_len) + "\t")
            f.write(str(self.q_max_len) + "\t")
            f.write(str(self.path_to_processed) + '\t')
            f.write(str(self.word_count) + '\t')

    def set_path(self):
        """
        set list of train, val, and test dataset paths

        Returns
            train_paths: list of train dataset paths for all task 1 to 20
            val_paths: list of val dataset paths for all task 1 to 20
            test_paths: list of test dataset paths for all task 1 to 20
        """
        train_paths = []
        val_paths = []
        test_paths = []

        contain = '.txt'
        for dirpath, dirnames, filenames in os.walk(self.path_to_babi):
            for filename in filenames:
                if 'train' in filename and contain in filename:
                    train_paths.append(os.path.join(dirpath, filename))
                elif 'val' in filename and contain in filename:
                    val_paths.append(os.path.join(dirpath, filename))
                elif 'test' in filename and contain in filename:
                    test_paths.append(os.path.join(dirpath, filename))
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

    def _split_paragraphs(self, path_to_file):
        """
        split into paragraphs as babi dataset consists of multiple 1~n sentences

        Args
            file_path: path of the data

        Returns
            paragraphs: list of paragraph
        """
        with open(path_to_file, 'r') as f:
            babi = f.readlines()
        paragraph = []
        paragraphs = []
        for d in babi:
            if d.startswith('1 '):
                if paragraph:
                    paragraphs.append(paragraph)
                paragraph = []
            paragraph.append(d)
        return paragraphs

    def _split_clqa(self, paragraphs, path_to_file, show_print=True):
        """
        for each paragraph, split into context, label, question and answer

        Args
            paragraphs: list of paragraphs
            path_to_file: path of the data

        Returns
            context: list of contexts
            label: list of labels
            question: list of questions
            answer: list of answers
        """
        context = []
        label = []
        question = []
        answer = []
        hint = []
        for paragraph in paragraphs:
            #            question_count = []
            for idx, sent in enumerate(paragraph):
                if '?' in sent:
                    alphabet = re.compile('[a-zA-Z]')
                    mark = re.search(alphabet, sent).span()[0]
                    q_a_ah = sent[mark:].split('\t')

                    question.append(q_a_ah[0].strip().lower())
                    answer.append(q_a_ah[1].strip().lower())
                    related_para = [para.strip().lower() for para in paragraph[:idx] if '?' not in para][::-1]
                    tmp_hint = []
                    for i, related_sent in enumerate(related_para):
                        if related_sent.split()[0] in q_a_ah[2].strip().split():
                            tmp_hint.append(i)
                    hint.append(tmp_hint)
                    if ("qa3" in path_to_file) and (len(related_para) > 130):
                        related_para = related_para[:130]
                    elif ("qa3" not in path_to_file) and (len(related_para) > 70):
                        related_para = related_para[:70]

                    context.append(related_para)
                    label.append([(i + 1) / 100 for i in range(len(related_para))])
        # update c_max_len
        for c in context:
            if len(c) > self.c_max_len:
                self.c_max_len = len(c)
        # check
        if show_print:
            if (len(question) == len(answer)) & (len(answer) == len(context)):
                print("bAbI is well separated into question, answer, and context!")
                print("total: {}".format(len(question)))
            else:
                print("Something is missing! check again")
                print("the number of questions: {}".format(len(question)))
                print("the number of answers: {}".format(len(answer)))
                print("the number of contexts: {}".format(len(context)))
                print("the number of labels: {}".format(len(label)))
        return context, label, question, answer, hint

    def split_all_clqa(self, paths, show_print=True):
        """
        merge all 20 babi tasks into one dataset

        Args:
            paths: list of path of 1 to 20 task dataset

        Returns:
            contexts: list of contexts of all 20 tasks
            labels: list of labels of all 20 tasks
            questions: list of questions of all 20 tasks
            answers: list of answers of all 20 tasks
        """
        if paths is None:
            print('path is None, run set_path() first!')
        else:
            contexts = []
            labels = []
            questions = []
            answers = []
            hints = []
            for path in paths:
                if show_print:
                    print('=================')
                paragraphs = self._split_paragraphs(path)
                if show_print:
                    print("data: {}".format(os.path.basename(path)))
                context, label, question, answer, hint = self._split_clqa(paragraphs, path, show_print=show_print)
                contexts.extend(context)
                labels.extend(label)
                questions.extend(question)
                answers.extend(answer)
                hints.extend(hint)
            return contexts, labels, questions, answers, hints

    def _set_word_set(self):
        c_word_set = set()
        q_word_set = set()
        a_word_set = set()
        train_context, train_label, train_question, train_answer, train_hint = self.split_all_clqa(self.train_paths,
                                                                                                   show_print=False)
        val_context, val_label, val_question, val_answer, val_hint = self.split_all_clqa(self.val_paths,
                                                                                         show_print=False)
        test_context, test_label, test_question, test_answer, test_hint = self.split_all_clqa(self.test_paths,
                                                                                              show_print=False)
        list_of_context = [train_context, val_context, test_context]
        list_of_question = [train_question, val_question, test_question]
        list_of_answer = [train_answer, val_answer, test_answer]
        for list_ in list_of_context:
            for para in list_:
                for sent in para:
                    sent = sent.replace(".", " .")
                    sent = sent.replace("?", " ?")
                    sent = sent.split()
                    c_word_set.update(sent[1:])
        for list_ in list_of_question:
            for sent in list_:
                sent = sent.replace(".", " .")
                sent = sent.replace("?", " ?")
                sent = sent.split()
                q_word_set.update(sent)
        for answers in list_of_answer:
            for answer in answers:
                answer = answer.split(',')
                a_word_set.update(answer)
        a_word_set.add(',')
        self._cqa_word_set = c_word_set.union(q_word_set).union(a_word_set)
        self.word_count = len(self._cqa_word_set)

    def _index_context(self, contexts):
        c_word_index = dict()
        for i, word in enumerate(self._cqa_word_set):
            c_word_index[word] = i + 1  # index 0 for zero padding
        indexed_cs = []
        for context in contexts:
            indexed_c = []
            for sentence in context:
                sentence = sentence.replace(".", " .")
                sentence = sentence.replace("?", " ?")
                sentence = sentence.split()
                indexed_s = []
                for word in sentence[1:]:
                    indexed_s.append(c_word_index[word])
                indexed_c.append(indexed_s)
            indexed_cs.append(np.array(indexed_c))
        return indexed_cs

    def _index_label(self, labels):
        indexed_ls = []
        for label in labels:
            indexed_ls.append(np.eye(self.c_max_len)[label])
        return indexed_ls

    def _index_question(self, questions):
        q_word_index = dict()
        for i, word in enumerate(self._cqa_word_set):
            q_word_index[word] = i + 1  # index 0 for zero padding
        indexed_qs = []
        for sentence in questions:
            sentence = sentence.replace(".", " .")
            sentence = sentence.replace("?", " ?")
            sentence = sentence.split()
            indexed_s = []
            for word in sentence:
                indexed_s.append(q_word_index[word])
            indexed_qs.append(np.array(indexed_s))
        return indexed_qs

    def _index_answer(self, answers):
        a_word_dict = dict()
        for i, word in enumerate(self._cqa_word_set):
            a_word_dict[word] = i
        indexed_as = []
        answer_num = []
        for answer in answers:
            indexed_a = np.zeros([3, len(self._cqa_word_set)], dtype=np.float32)
            answer_bool = np.zeros(3, dtype=np.float32)
            for i, a in enumerate(answer.split(',')):
                indexed_a[i, a_word_dict[a]] = 1
                answer_bool[i] = 1
            indexed_as.append(indexed_a)
            answer_num.append(answer_bool)

        if not os.path.exists(self.path_to_processed):
            os.makedirs(self.path_to_processed)

        with open(os.path.join(self.path_to_processed, 'answer_word_dict.pkl'), 'wb') as f:
            pickle.dump(a_word_dict, f)
        return indexed_as, answer_num

    def masking(self, context_index, label_index, question_index):
        context_masked = []
        question_masked = []
        label_masked = []
        context_real_len = []
        question_real_len = []
        # cs: one context
        for cs, l, q in zip(context_index, label_index, question_index):
            context_masked_tmp = []
            context_real_length_tmp = []
            # cs: many sentences
            for context in cs:
                context_real_length_tmp.append(len(context))
                diff = self.s_max_len - len(context)
                if diff > 0:
                    context_mask = np.append(context, [self.mask_index] * diff, axis=0)
                    context_masked_tmp.append(context_mask.tolist())
                else:
                    context_masked_tmp.append(context)
            diff_c = self.c_max_len - len(cs)
            context_masked_tmp.extend([[0] * self.s_max_len] * diff_c)
            context_masked.append(context_masked_tmp)
            context_real_length_tmp.extend([0] * diff_c)
            context_real_len.append(context_real_length_tmp)

            diff_q = self.q_max_len - len(q)
            question_real_len.append(len(q))
            question_masked_tmp = np.array(np.append(q, [self.mask_index] * diff_q, axis=0))
            question_masked.append(question_masked_tmp.tolist())

            diff_l = self.c_max_len - len(l)
            label_masked_tmp = np.append(l, [self.mask_index] * diff_l, axis=0)
            label_masked.append(label_masked_tmp.tolist())
        return context_masked, question_masked, label_masked, context_real_len, question_real_len

    def load_train(self):
        train_context, train_label, train_question, train_answer, train_hint = self.split_all_clqa(self.train_paths)
        train_context_index = self._index_context(train_context)
        train_label_index = train_label
        train_question_index = self._index_question(train_question)
        train_answer_index, train_answer_num = self._index_answer(train_answer)
        # check max sentence length
        for context in train_context_index:
            for sentence in context:
                if len(sentence) > self.s_max_len:
                    self.s_max_len = len(sentence)
        # check max question length
        for question in train_question_index:
            if len(question) > self.q_max_len:
                self.q_max_len = len(question)
        train_context_masked, train_question_masked, train_label_masked, train_context_real_len, train_question_real_len = self.masking(
            train_context_index, train_label_index, train_question_index)
        # check masking
        cnt = 0
        for c, q, l in zip(train_context_masked, train_question_masked, train_label_masked):
            for context in c:
                if (len(context) != self.s_max_len) | (len(q) != self.q_max_len):
                    cnt += 1
        if cnt == 0:
            print("Train Masking success!")
        else:
            print("Train Masking process error")
        train_dataset = (
            train_question_masked, train_answer_index, train_answer_num, train_context_masked, train_label_masked,
            train_context_real_len, train_question_real_len, train_hint)
        if not os.path.exists(self.path_to_processed):
            os.makedirs(self.path_to_processed)
        with open(os.path.join(self.path_to_processed, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)

    def load_val(self):
        val_context, val_label, val_question, val_answer, val_hint = self.split_all_clqa(self.val_paths)
        val_context_index = self._index_context(val_context)
        val_label_index = val_label
        val_question_index = self._index_question(val_question)
        val_answer_index, val_answer_num = self._index_answer(val_answer)
        val_context_masked, val_question_masked, val_label_masked, val_context_real_len, val_question_real_len = self.masking(
            val_context_index, val_label_index, val_question_index)
        # check masking
        cnt = 0
        for c, q, l in zip(val_context_masked, val_question_masked, val_label_masked):
            for context in c:
                if (len(context) != self.s_max_len) | (len(q) != self.q_max_len):
                    cnt += 1
        if cnt == 0:
            print("Val Masking success!")
        else:
            print("Val Masking process error")
        val_dataset = (val_question_masked, val_answer_index, val_answer_num, val_context_masked, val_label_masked,
                       val_context_real_len, val_question_real_len, val_hint)
        if not os.path.exists(self.path_to_processed):
            os.makedirs(self.path_to_processed)
        with open(os.path.join(self.path_to_processed, 'val_dataset.pkl'), 'wb') as f:
            pickle.dump(val_dataset, f)

    def load_test(self):
        test_context, test_label, test_question, test_answer, test_hint = self.split_all_clqa(self.test_paths)
        with open(os.path.join(self.path_to_processed, 'test_context.pkl'), 'wb') as f:
            pickle.dump(test_context, f)
        with open(os.path.join(self.path_to_processed, 'test_question.pkl'), 'wb') as f:
            pickle.dump(test_question, f)
        with open(os.path.join(self.path_to_processed, 'test_answer.pkl'), 'wb') as f:
            pickle.dump(test_answer, f)
        test_context_index = self._index_context(test_context)
        test_label_index = test_label
        test_question_index = self._index_question(test_question)
        test_answer_index, test_answer_num = self._index_answer(test_answer)
        test_context_masked, test_question_masked, test_label_masked, test_context_real_len, test_question_real_len = self.masking(
            test_context_index, test_label_index, test_question_index)
        # check masking
        cnt = 0
        for c, q, l in zip(test_context_masked, test_question_masked, test_label_masked):
            for context in c:
                if (len(context) != self.s_max_len) | (len(q) != self.q_max_len):
                    cnt += 1
        if cnt == 0:
            print("Test Masking success!")
        else:
            print("Test Masking process error")
        test_dataset = (
            test_question_masked, test_answer_index, test_answer_num, test_context_masked, test_label_masked,
            test_context_real_len, test_question_real_len, test_hint)
        if not os.path.exists(self.path_to_processed):
            os.makedirs(self.path_to_processed)
        with open(os.path.join(self.path_to_processed, 'test_dataset.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)

    ### dialog preprocessing ###

    def run_dialog(self):
        trn_context, trn_context_real_len, trn_question, trn_question_real_len, trn_match_words, trn_answer = self.vectorize_data(
            self.train)
        val_context, val_context_real_len, val_question, val_question_real_len, val_match_words, val_answer = self.vectorize_data(
            self.val)
        tst_context, tst_context_real_len, tst_question, tst_question_real_len, tst_match_words, tst_answer = self.vectorize_data(
            self.test)
        candidate = self.vectorize_candidates(self.candidates)

        trn_dataset = (
            trn_context, trn_context_real_len, trn_question, trn_question_real_len, trn_match_words, trn_answer)
        val_dataset = (
            val_context, val_context_real_len, val_question, val_question_real_len, val_match_words, val_answer)
        tst_dataset = (
            tst_context, tst_context_real_len, tst_question, tst_question_real_len, tst_match_words, tst_answer)
        candidate_dataset = candidate

        ### save to pickle
        log.infov('save to {} ...'.format(self.path_to_processed))
        with open(os.path.join(self.path_to_processed, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(trn_dataset, f)
        with open(os.path.join(self.path_to_processed, 'val_dataset.pkl'), 'wb') as f:
            pickle.dump(val_dataset, f)
        with open(os.path.join(self.path_to_processed, 'test_dataset.pkl'), 'wb') as f:
            pickle.dump(tst_dataset, f)
        with open(os.path.join(self.path_to_processed, 'idx_to_word.pkl'), 'wb') as f:
            pickle.dump(self.idx2w, f)
        with open(os.path.join(self.path_to_processed, 'word_to_idx.pkl'), 'wb') as f:
            pickle.dump(self.w2idx, f)
        with open(os.path.join(self.path_to_processed, 'idx_to_cand.pkl'), 'wb') as f:
            pickle.dump(self.idx2candid, f)
        with open(os.path.join(self.path_to_processed, 'cand_to_idx.pkl'), 'wb') as f:
            pickle.dump(self.candid2idx, f)
        with open(os.path.join(self.path_to_processed, 'cand_set.pkl'), 'wb') as f:
            pickle.dump(candidate_dataset, f)

        ### write data info to config_taskID.txt
        if self.is_oov:
            with open(os.path.join('./babi_dialog', 'config_oov_' + self.task + '.txt'), 'w') as f:
                f.write(str(self.memory_size) + '\t')
                f.write(str(self.sentence_size) + '\t')
                f.write(str(self.candidate_sentence_size) + '\t')
                f.write(str(self.n_cand) + '\t')
                f.write(str(self.vocab_size) + '\t')
                f.write(str(self.path_to_processed) + '\t')
                f.write(str(_TYPE_NUM) + '\t')
        else:
            with open(os.path.join('./babi_dialog', 'config_' + self.task + '.txt'), 'w') as f:
                f.write(str(self.memory_size) + '\t')
                f.write(str(self.sentence_size) + '\t')
                f.write(str(self.candidate_sentence_size) + '\t')
                f.write(str(self.n_cand) + '\t')
                f.write(str(self.vocab_size) + '\t')
                f.write(str(self.path_to_processed) + '\t')
                f.write(str(_TYPE_NUM) + '\t')

    def vectorize_data(self, data):
        S = []
        S_real_lens = []
        Q = []
        Q_real_lens = []
        SQ_match_word = []
        A = []
        for i, (story, query, answer) in tqdm(enumerate(data)):
            memory_size = self.memory_size
            ss = []
            s_lens = []
            cuisine = []
            location = []
            price = []
            rating = []
            phone = []
            address = []
            number = []
            match_word = [[]] * _TYPE_NUM
            for _, sentence in enumerate(story, 1):
                ls = max(0, self.sentence_size - len(sentence))
                ss.append([self.w2idx[w] if w in self.w2idx else 0 for w in sentence] + [0] * ls)
                s_lens.append(len(sentence))
                for w in sentence:
                    if w in _CUISINE:
                        cuisine.append(w)
                    elif w in _LOCATION:
                        location.append(w)
                    elif w in _PRICE:
                        price.append(w)
                    elif w in _RATING:
                        rating.append(w)
                    elif w in _PHONE:
                        phone.append(w)
                    elif w in _ADDRESS:
                        address.append(w)
                    elif w in _NUMBER:
                        number.append(w)

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * self.sentence_size)
                s_lens.append(0)

            lq = max(0, self.sentence_size - len(query))
            q = [self.w2idx[w] if w in self.w2idx else 0 for w in query] + [0] * lq
            q_lens = len(query)
            for w in query:
                if w in _CUISINE:
                    cuisine.append(w)
                elif w in _LOCATION:
                    location.append(w)
                elif w in _PRICE:
                    price.append(w)
                elif w in _RATING:
                    rating.append(w)
                elif w in _PHONE:
                    phone.append(w)
                elif w in _ADDRESS:
                    address.append(w)
                elif w in _NUMBER:
                    number.append(w)

            if cuisine:
                match_word[0] = [self.w2idx[cuisine[-1]]]
            if location:
                match_word[1] = [self.w2idx[location[-1]]]
            if price:
                match_word[2] = [self.w2idx[price[-1]]]
            if rating:
                match_word[3] = [self.w2idx[rating[-1]]]
            if phone:
                match_word[4] = [self.w2idx[phone[-1]]]
            if address:
                match_word[5] = [self.w2idx[address[-1]]]
            if number:
                match_word[6] = [self.w2idx[number[-1]]]

            S.append(np.array(ss))
            S_real_lens.append(np.array(s_lens))
            Q.append(np.array(q))
            Q_real_lens.append(np.array(q_lens))
            SQ_match_word.append(match_word)
            A.append(np.array(answer))
        return S, S_real_lens, Q, Q_real_lens, SQ_match_word, A

    def vectorize_candidates(self, candidates):
        C = []
        for i, candidate in tqdm(enumerate(candidates)):
            lc = max(0, self.candidate_sentence_size - len(candidate))
            C.append([self.w2idx[w] if w in self.w2idx else 0 for w in candidate] + [0] * lc)
        return C


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='story', help='story/dialog')
    config = parser.parse_args()
    preprocess = Preprocess(config)
