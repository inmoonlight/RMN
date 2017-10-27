import numpy as np
import pandas as pd
import pickle
import os

import argparse
import tensorflow as tf
from utils import read_data, batch_iter_test, parse_config, read_data_test

dir(tf.contrib)

flags = tf.app.flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '--version')
    parser.add_argument('--gpu_frac', type=float, default=0.95)
    parser.add_argument('--metagraph', '--metagraph')
    _version = parser.parse_args().version  # 8_1
    _gpu_frac = parser.parse_args().gpu_frac
    _meta = parser.parse_args().metagraph

    global config
    with open('config_-1.txt', 'r') as f:
        config = parse_config(f.readline())
    checkpoint = '/'.join(_meta.split('/')[:-1])

    [test_q, test_a, test_a_num, test_c, test_l, test_c_real_len, test_q_real_len,
     test_hint], word_set = read_data_test(
        config['babi_processed'])
    num = len(test_c)
    batch_size = config['batch_size']
    crcts = []
    preds = []
    alphas_1 = []
    alphas_2 = []
    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess_config.gpu_options.per_process_gpu_memory_fraction = _gpu_frac
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            saver = tf.train.import_meta_graph(_meta)
            print("checkpoint: {}".format(tf.train.latest_checkpoint(checkpoint)))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            graph = tf.get_default_graph()
            # placeholder
            context = graph.get_tensor_by_name("context:0")
            sentence_real_len = graph.get_tensor_by_name("sentence_real_length:0")
            question = graph.get_tensor_by_name("question:0")
            question_real_len = graph.get_tensor_by_name("question_real_length:0")
            label = graph.get_tensor_by_name("label:0")
            answer = graph.get_tensor_by_name("answer:0")
            answer_num = graph.get_tensor_by_name("answer_num:0")
            is_training = graph.get_tensor_by_name("is_training:0")
            hint = graph.get_tensor_by_name("hint:0")

            assert sess.graph == graph

            correct = graph.get_collection("correct")[0]
            prediction = graph.get_collection("final_pred")[0]
            alpha_1 = graph.get_tensor_by_name("a_1/alpha:0")
            alpha_2 = graph.get_tensor_by_name("a_2/alpha:0")
            # alpha_1 = graph.get_collection('alpha_1')[0]
            # alpha_2 = graph.get_collection('alpha_2')[0]
            # loss = graph.get_collection("loss")[0]
            # accuracy = graph.get_collection("accuracy")[0]

            batch_test = batch_iter_test(c=test_c, q=test_q, l=test_l, a=test_a, a_num=test_a_num,
                                         c_real_len=test_c_real_len, q_real_len=test_q_real_len, h=test_hint,
                                         batch_size=config['batch_size'], num_epochs=1, _version=_version,
                                         shuffle=False, c_max_len = config['c_max_len'])
            for test in batch_test:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch, h_batch = zip(
                    *test)
                feed_dict = {context: c_batch,
                             question: q_batch,
                             label: l_batch,
                             answer: a_batch,
                             answer_num: a_num_batch,
                             sentence_real_len: c_real_len_batch,
                             question_real_len: q_real_len_batch,
                             hint: h_batch,
                             is_training: False}
                correct_test, pred_test, alpha_test_1, alpha_test_2 = sess.run([correct, prediction, alpha_1, alpha_2], feed_dict=feed_dict)
                crcts.append(correct_test)
                preds.append(pred_test)
                alphas_1.append(alpha_test_1)
                alphas_2.append(alpha_test_2)

    result_c = []
    result_p = []
    result_a_1 = []
    result_a_2 = []
    for crct, pred, alp_1, alp_2 in zip(crcts, preds, alphas_1, alphas_2):
        for c in crct:
            result_c.append(c)
        for p in pred:
            result_p.append(np.where(p == 1)[0])
        for a1 in alp_1:
            result_a_1.append(a1[0])
        for a2 in alp_2:
            result_a_2.append(a2[0])

    # make result table
    # TODO: add real hint, model hint

    reversed_word_set = {v: k for k, v in word_set.items()}
    # context
    original_c = []
    for context in test_c:
        original_s = []
        for sentence in context:
            while 0 in sentence:
                sentence.remove(0)
            sentence = [reversed_word_set[word_index - 1] for word_index in sentence]
            original_s.append(" ".join(sentence).strip())
        original_c.append("@".join(original_s).strip())
    # question
    original_q = []
    for question in test_q:
        while 0 in question:
            question.remove(0)
        question = [reversed_word_set[word_index - 1] for word_index in question]
        original_q.append(" ".join(question).strip())
    # answer
    original_a = []
    for answers, nums in zip(test_a, test_a_num):
        answer_indices = list(np.argmax(answers, axis=1) * nums.ravel())
        while 0 in answer_indices:
            answer_indices.remove(0)
        answer = [reversed_word_set[word_index] for word_index in answer_indices]
        original_a.append(", ".join(answer).strip())
    # model_answer
    model_a = []
    for a_index in result_p:
        model_answer = [reversed_word_set[word_index] for word_index in list(a_index)]
        model_a.append(", ".join(model_answer).strip())
    # model attention coeff
    model_alpha_1 = []
    for a1_coff in result_a_1:
        model_a1 = [str(a1_att) for a1_att in list(a1_coff)]
        model_alpha_1.append("@".join(model_a1).strip())
    model_alpha_2 = []
    for a2_coff in result_a_2:
        model_a2 = [str(a2_att) for a2_att in list(a2_coff)]
        model_alpha_2.append("@".join(model_a2).strip())




    result = pd.DataFrame(columns=['Context','alpha_1','alpha_2', 'Question', 'Answer', 'Model_Answer', 'score'])
    result['Context'] = original_c
    result['Question'] = original_q
    result['Answer'] = original_a
    # batch
    model_a = model_a[: (num//batch_size) * batch_size] + model_a[-(num%batch_size):]
    result_c = result_c[:(num//batch_size)*batch_size] + result_c[-(num%batch_size):]
    result['Model_Answer'] = model_a
    result['score'] = result_c
    result['alpha_1'] = model_alpha_1[: (num//batch_size)*batch_size] + model_alpha_1[-(num%batch_size):]
    result['alpha_2'] = model_alpha_2[: (num//batch_size)*batch_size] + model_alpha_2[-(num%batch_size):]

    tasks = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 2, 3, 4, 5, 6, 7, 8, 9]
    splits = np.cumsum(
        [0, 995, 995, 995, 995, 995, 996, 999, 992, 997, 999, 995, 993, 995, 995, 999, 995, 995, 995, 995, 995])
    splited_result = [result[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
    for task_result, task_num in zip(splited_result, tasks):
        print("task {0} acc: {1:.4f} %".format(task_num, (sum(task_result.score) / len(task_result) * 100)))
        visual_c, visual_a1, visual_a2 = [], [], []
        for vc, va1, va2 in zip(task_result.Context.apply(lambda x: x.split("@")), task_result.alpha_1.apply(lambda x: [float(y) for y in x.split("@")]), task_result.alpha_2.apply(lambda x: [float(y) for y in x.split("@")])):
            visual_c.extend(vc)
            visual_a1.extend(va1)
            visual_a2.extend(va2)
        final_result = task_result[task_result.columns[3:]].loc[np.repeat(task_result.index, [config['c_max_len']]*task_result.shape[0])]
        final_result['Context'] = visual_c
        final_result['alpha_1'] = visual_a1
        final_result['alpha_2'] = visual_a2
        final_result.to_csv(os.path.join(checkpoint, 'task_{}.csv'.format(task_num)), index = False)

if __name__ == '__main__':
    main()
