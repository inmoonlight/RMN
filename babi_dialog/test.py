import argparse
import numpy as np
import pandas as pd
import sys

sys.path.append('../../')

import tensorflow as tf

from input_ops import load_data, batch_iter_test
from util import log


def run(config):
    task_idx = config.task
    is_oov = config.is_oov
    gpu_frac = config.gpu_frac
    metagraph = config.metagraph
    checkpoint = '/'.join(metagraph.split('/')[:-1])

    batch_size = config.batch_size

    if is_oov:
        task_idx = 'oov_{}'.format(task_idx)

    # load test data
    test_dataset = load_data(task=task_idx, type='test')
    candidate = load_data(task=task_idx, type='candidate')
    idx_to_cand = load_data(task=task_idx, type='idx_to_cand')
    idx_to_word = load_data(task=task_idx, type='idx_to_word')

    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        sess = tf.Session(config=sess_config)

        crcts = []
        max_score_idxs = []
        answer_idxs = []
        alphas_1 = []
        alphas_2 = []
        with sess.as_default():
            saver = tf.train.import_meta_graph(metagraph)
            log.warning("checkpoint: {}".format(tf.train.latest_checkpoint(checkpoint)))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            graph = tf.get_default_graph()
            assert sess.graph == graph

            context = graph.get_tensor_by_name("context:0")
            # context_real_len = graph.get_tensor_by_name("context_real_length:0")
            question = graph.get_tensor_by_name("question:0")
            # question_real_len = graph.get_tensor_by_name("question_real_length:0")
            answer = graph.get_tensor_by_name("answer:0")
            # answer_match = graph.get_tensor_by_name("answer_match:0")
            answer_idx = graph.get_tensor_by_name("answer_idx:0")
            is_training = graph.get_tensor_by_name("is_training:0")

            pred = graph.get_collection("prediction")[0]
            max_score_idx = graph.get_collection("max_score_idx")[0]
            a_1 = graph.get_tensor_by_name("a_1/alpha:0")
            try:
                a_2 = graph.get_tensor_by_name("a_2/alpha:0")
            except:
                pass
            correct = graph.get_collection("correct")[0]
            accuracy = graph.get_collection("accuracy")[0]
            loss = graph.get_collection("loss")[0]

            batch_test = batch_iter_test(c=test_dataset[0],
                                         c_real_len=test_dataset[1],
                                         q=test_dataset[2],
                                         q_real_len=test_dataset[3],
                                         cand=candidate,
                                         match_words=test_dataset[4],
                                         a_idx=test_dataset[5],
                                         batch_size=batch_size)

            for test in batch_test:
                c_batch, c_real_len_batch, q_batch, q_real_len_batch, cand_batch, cand_match_batch, a_idx_batch = zip(
                    *test)
                feed_dict = {context: c_batch,
                             # context_real_len: c_real_len_batch,
                             question: q_batch,
                             # question_real_len: q_real_len_batch,
                             answer: cand_batch,
                             # answer_match: cand_match_batch,
                             answer_idx: a_idx_batch,
                             is_training: False}
                try:
                    correct_test, max_score_idx_test, alpha_1, alpha_2, answer_idx_test = sess.run(
                        [correct, max_score_idx, a_1, a_2, answer_idx], feed_dict=feed_dict)
                except:
                    correct_test, max_score_idx_test, alpha_1, answer_idx_test = sess.run(
                        [correct, max_score_idx, a_1, answer_idx], feed_dict=feed_dict)
                crcts.extend(correct_test)
                max_score_idxs.extend(max_score_idx_test)
                answer_idxs.extend(answer_idx_test)
                alphas_1.append(alpha_1)
                try:
                    alphas_2.append(alpha_2)
                except:
                    pass

                if config.print:
                    model_ans_idxs = [ans_idx for ans_idx in max_score_idx_test[:1]]
                    ans_idxs = [idx for idx in a_idx_batch[:1]]

                    for c, a1, model_ans_idx, ans_idx in zip(c_batch[:1], np.squeeze(alpha_1)[:1],
                                                             model_ans_idxs, ans_idxs):
                        for idx in (-a1).argsort()[:10]:
                            prob = "{0:2f}".format(a1[idx])
                            ss = [prob]
                            for c_word_idx in c[idx]:
                                if c_word_idx == 0:
                                    ss.append("")
                                else:
                                    ss.append(idx_to_word[c_word_idx - 1])
                            log.critical(" ".join(ss))

                        print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                        print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                        print(model_ans_idx, ans_idx)

        result_a = []
        for alp in alphas_1:
            for a1 in alp:
                c_max_len = len(a1[0])
                result_a.append(a1[0])

        result_a2 = []
        try:
            for alp in alphas_2:
                for a2 in alp:
                    c_max_len = len(a2[0])
                    result_a2.append(a2[0])
        except:
            pass

        data_size = len(test_dataset[1])
        batch_num = data_size // batch_size
        left_over = data_size % batch_size
        crcts_rm_overlap = crcts[:batch_size * batch_num] + crcts[-left_over:]
        max_score_idxs_rm_overlap = max_score_idxs[:batch_size * batch_num] + max_score_idxs[-left_over:]
        answer_idxs_rm_overlap = answer_idxs[:batch_size * batch_num] + answer_idxs[-left_over:]
        alpha_overlap = result_a[:batch_size * batch_num] + result_a[-left_over:]
        try:
            alpha2_overlap = result_a2[:batch_size * batch_num] + result_a2[-left_over:]
        except:
            pass

        log.infov("Test acc: {}".format(sum(crcts_rm_overlap) / len(crcts_rm_overlap)))

        original_c = []
        for context in test_dataset[0]:
            original_s = []
            for sentence in context:
                sentence = list(sentence)
                while 0 in sentence:
                    sentence.remove(0)
                sentence = [idx_to_word[word_index - 1] for word_index in sentence]
                original_s.append(" ".join(sentence).strip())
            original_c.append("@".join(original_s).strip())

        original_q = []
        for question in test_dataset[2]:
            question = list(question)
            while 0 in question:
                question.remove(0)
            question = [idx_to_word[word_index - 1] for word_index in question]
            original_q.append(' '.join(question).strip())

        original_a = []
        for aidx in answer_idxs_rm_overlap:
            original_a.append(idx_to_cand[aidx])

        original_p = []
        for msidx in max_score_idxs_rm_overlap:
            original_p.append(idx_to_cand[msidx])

        original_alpha = []
        for alp_over in alpha_overlap:
            model_alpha = [str(a_att) for a_att in list(alp_over)]
            original_alpha.append("@".join(model_alpha).strip())

        original_alpha2 = []
        try:
            for alp_over in alpha2_overlap:
                model_alpha = [str(a_att) for a_att in list(alp_over)]
                original_alpha2.append("@".join(model_alpha).strip())
        except:
            pass

        if len(original_alpha2) > 0:
            result = pd.DataFrame(columns=['Context', 'alpha1', 'alpha2', 'question', 'answer', 'model_answer', 'correct'])
            result['Context'] = original_c
            result['question'] = original_q
            result['alpha1'] = original_alpha
            result['alpha2'] = original_alpha2
            result['answer'] = original_a
            result['model_answer'] = original_p
            result['correct'] = crcts_rm_overlap

            visual_c, visual_alp1, visual_alp2 = [], [], []
            for vc, va1, va2 in zip(result.Context.apply(lambda x: x.split("@")),
                              result.alpha1.apply(lambda x: [float(y) for y in x.split("@")]),
                              result.alpha2.apply(lambda x: [float(y) for y in x.split("@")])):
                visual_c.extend(vc)
                visual_alp1.extend(va1)
                visual_alp2.extend(va2)

            final_result = result[result.columns[2:]].loc[np.repeat(result.index, [c_max_len] * result.shape[0])]
            final_result['Context'] = visual_c
            final_result['alpha1'] = visual_alp1
            final_result['alpha2'] = visual_alp2
            final_result.to_csv(checkpoint + '/result.csv')

        else:
            result = pd.DataFrame(columns=['Context', 'alpha', 'question', 'answer', 'model_answer', 'correct'])
            result['Context'] = original_c
            result['question'] = original_q
            result['alpha'] = original_alpha
            result['answer'] = original_a
            result['model_answer'] = original_p
            result['correct'] = crcts_rm_overlap

            visual_c, visual_alp = [], []
            for vc, va in zip(result.Context.apply(lambda x: x.split("@")),
                              result.alpha.apply(lambda x: [float(y) for y in x.split("@")])):
                visual_c.extend(vc)
                visual_alp.extend(va)

            final_result = result[result.columns[2:]].loc[np.repeat(result.index, [c_max_len] * result.shape[0])]
            final_result['Context'] = visual_c
            final_result['alpha'] = visual_alp
            final_result.to_csv(checkpoint + '/result.csv')


# for answer_idx, max_score_idx, crct in zip(answer_idxs_rm_overlap, max_score_idxs_rm_overlap, crcts_rm_overlap):
#            result_dict['answer'].append(idx_to_cand[answer_idx])
#            result_dict['model_answer'].append(idx_to_cand[max_score_idx])
#            result_dict['correct'].append(crct)

#        result = pd.DataFrame(result_dict)
#        result.to_csv(checkpoint + '/result.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metagraph', '--metagraph')
    parser.add_argument('--task', type=int)
    parser.add_argument('--is_oov', type=bool, default=False)
    parser.add_argument('--gpu_frac', type=float, default=0.45)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--print', type=bool, default=False)
    config = parser.parse_args()

    run(config)
