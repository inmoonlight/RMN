import argparse
import numpy as np
import os
import sys
from time import time

sys.path.append('../../')

import tensorflow as tf

import alarm
from input_ops import load_data, batch_iter
from util import log

alarm_channel = 'babi_dialog'


def run(config):
    task_idx = config.task
    is_oov = config.is_oov
    gpu_frac = config.gpu_frac
    metagraph = config.metagraph
    display_step = config.display_step
    checkpoint = '/'.join(metagraph.split('/')[:-1])
    version = metagraph.split('/')[-2]
    save_dir = "/".join(metagraph.split("/")[:-1])
    save_summary_path = os.path.join(save_dir, 'model_summary')
    save_variable_path = os.path.join(save_dir, 'model_variables')

    batch_size = config.batch_size
    iter_time = config.iter_time

    if is_oov:
        task_idx = 'oov_{}'.format(task_idx)

    train_dataset = load_data(task=task_idx, type='train')
    if config.use_test:  # TODO: Remove when upload to the public github
        val_dataset = load_data(task=task_idx, type='test')
    else:
        val_dataset = load_data(task=task_idx, type='val')
    candidate = load_data(task=task_idx, type='candidate')
    idx_to_cand = load_data(task=task_idx, type='idx_to_cand')
    idx_to_word = load_data(task=task_idx, type='idx_to_word')

    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        sess = tf.Session(config=sess_config)
        max_acc = 0
        min_loss = 200
        alarm.send_message("Start retraining task_{} ...".format(task_idx), channel=alarm_channel)
        start_time = time()
        with sess.as_default():
            saver = tf.train.import_meta_graph(metagraph)
            log.warning("checkpoint: {}".format(tf.train.latest_checkpoint(checkpoint)))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            graph = tf.get_default_graph()
            assert sess.graph == graph

            ## placeholder
            context = graph.get_tensor_by_name("context:0")
            context_real_len = graph.get_tensor_by_name("context_real_length:0")
            question = graph.get_tensor_by_name("question:0")
            question_real_len = graph.get_tensor_by_name("question_real_length:0")
            answer = graph.get_tensor_by_name("answer:0")
            answer_match = graph.get_tensor_by_name("answer_match:0")
            answer_idx = graph.get_tensor_by_name("answer_idx:0")
            is_training = graph.get_tensor_by_name("is_training:0")

            ## training parameters
            global_step = graph.get_tensor_by_name("global_step:0")
            optimizer = tf.get_collection("optimizer")[0]
            loss = tf.get_collection("loss")[0]
            accuracy = tf.get_collection("accuracy")[0]

            ## model variable
            if 'hop4' in version:
                a_1 = graph.get_tensor_by_name("a_1/alpha:0")
                a_2 = graph.get_tensor_by_name("a_2/alpha:0")
                a_3 = graph.get_tensor_by_name("a_3/alpha:0")
                a_4 = graph.get_tensor_by_name("a_4/alpha:0")
            elif 'hop3' in version:
                a_1 = graph.get_tensor_by_name("a_1/alpha:0")
                a_2 = graph.get_tensor_by_name("a_2/alpha:0")
                a_3 = graph.get_tensor_by_name("a_3/alpha:0")
            elif 'hop2' in version:
                a_1 = graph.get_tensor_by_name("a_1/alpha:0")
                a_2 = graph.get_tensor_by_name("a_2/alpha:0")
            elif 'hop1' in version:
                a_1 = graph.get_tensor_by_name("a_1/alpha:0")
            max_score_idx = tf.get_collection("max_score_idx")[0]

            loss_train = tf.summary.scalar("loss_train", loss)
            accuracy_train = tf.summary.scalar("accuracy_train", accuracy)
            train_summary_ops = tf.summary.merge([loss_train, accuracy_train])
            loss_val = tf.summary.scalar("loss_val", loss)
            accuracy_val = tf.summary.scalar("accuracy_val", accuracy)
            val_summary_ops = tf.summary.merge([loss_val, accuracy_val])
            summary_writer = tf.summary.FileWriter(save_summary_path, graph)

            batch_train = batch_iter(task_idx=task_idx,
                                     c=train_dataset[0],
                                     c_real_len=train_dataset[1],
                                     q=train_dataset[2],
                                     q_real_len=train_dataset[3],
                                     cand=candidate,
                                     match_words=train_dataset[4],
                                     a_idx=train_dataset[5],
                                     shuffle=True,
                                     batch_size=batch_size,
                                     num_epochs=iter_time,
                                     version=version,
                                     is_alarm=True)
            for train in batch_train:
                c_batch, c_real_len_batch, q_batch, q_real_len_batch, cand_batch, cand_match_batch, a_idx_batch = zip(
                    *train)
                feed_dict = {context: c_batch,
                             context_real_len: c_real_len_batch,
                             question: q_batch,
                             question_real_len: q_real_len_batch,
                             answer: cand_batch,
                             answer_match: cand_match_batch,
                             answer_idx: a_idx_batch,
                             is_training: True}
                current_step = sess.run(global_step, feed_dict=feed_dict)
                optimizer.run(feed_dict=feed_dict)
                train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
                summary_writer.add_summary(train_summary, current_step)
                if current_step % display_step == 0:
                    log.warning("train_loss: {} / train_acc: {}".format(sess.run(loss, feed_dict=feed_dict),
                                                                        sess.run(accuracy, feed_dict=feed_dict)))
                    print("step: {}".format(current_step))
                    # print model answer and real answer
                    if 'hop4' is version:
                        model_ans_idx, alpha_1, alpha_2, alpha_3, alpha_4 = sess.run(
                            [max_score_idx, a_1, a_2, a_3, a_4], feed_dict=feed_dict)
                    elif 'hop3' in version:
                        model_ans_idx, alpha_1, alpha_2, alpha_3 = sess.run(
                            [max_score_idx, a_1, a_2, a_3], feed_dict=feed_dict)
                    elif 'hop2' in version:
                        model_ans_idx, alpha_1, alpha_2 = sess.run(
                            [max_score_idx, a_1, a_2], feed_dict=feed_dict)
                    elif 'hop1' in version:
                        model_ans_idx, alpha_1 = sess.run(
                            [max_score_idx, a_1], feed_dict=feed_dict)
                    else:
                        model_ans_idx = sess.run([max_score_idx], feed_dict=feed_dict)
                    model_ans_idxs = [ans_idx for ans_idx in model_ans_idx[:3]]
                    ans_idxs = [idx for idx in a_idx_batch[:3]]

                    if 'hop2' in version:
                        for c, a1, a2, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
                                                                     np.squeeze(alpha_2)[:3],
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

                            print("\n")

                            for idx in (-a2).argsort()[:10]:
                                prob = "{0:2f}".format(a2[idx])
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
                            print("===============================================")

                    elif 'hop1' in version:
                        for c, a1, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
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
                            print("===============================================")

                    print("==validation==")
                    batch_val = batch_iter(task_idx=task_idx,
                                           c=val_dataset[0],
                                           c_real_len=val_dataset[1],
                                           q=val_dataset[2],
                                           q_real_len=val_dataset[3],
                                           cand=candidate,
                                           match_words=val_dataset[4],
                                           a_idx=val_dataset[5],
                                           shuffle=True,
                                           batch_size=batch_size,
                                           num_epochs=1,
                                           version=version,
                                           is_alarm=False)
                    accs = []
                    losses = []
                    print_only_first = True
                    print("===============================================")
                    for val in batch_val:
                        c_val, c_real_len_val, q_val, q_real_len_val, cand_val, cand_match_val, a_idx_val = zip(*val)
                        feed_dict = {context: c_val,
                                     context_real_len: c_real_len_val,
                                     question: q_val,
                                     question_real_len: q_real_len_val,
                                     answer: cand_val,
                                     answer_match: cand_match_val,
                                     answer_idx: a_idx_val,
                                     is_training: False}
                        # print model answer and real answer
                        if print_only_first:

                            if 'hop4' is version:
                                model_ans_idx, alpha_1, alpha_2, alpha_3, alpha_4 = sess.run(
                                    [max_score_idx, a_1, a_2, a_3, a_4], feed_dict=feed_dict)
                            elif 'hop3' in version:
                                model_ans_idx, alpha_1, alpha_2, alpha_3 = sess.run(
                                    [max_score_idx, a_1, a_2, a_3], feed_dict=feed_dict)
                            elif 'hop2' in version:
                                model_ans_idx, alpha_1, alpha_2 = sess.run(
                                    [max_score_idx, a_1, a_2], feed_dict=feed_dict)
                            elif 'hop1' in version:
                                model_ans_idx, alpha_1 = sess.run(
                                    [max_score_idx, a_1], feed_dict=feed_dict)
                            else:
                                model_ans_idx = sess.run([max_score_idx], feed_dict=feed_dict)
                            model_ans_idxs = [ans_idx for ans_idx in model_ans_idx[:3]]
                            ans_idxs = [idx for idx in a_idx_batch[:3]]

                            if 'hop2' in version:
                                for c, a1, a2, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
                                                                             np.squeeze(alpha_2)[:3],
                                                                             model_ans_idxs, ans_idxs):
                                    for idx in (-a1).argsort()[:10]:
                                        prob = "{0:2f}".format(a1[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a2).argsort()[:10]:
                                        prob = "{0:2f}".format(a2[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                                    print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                                    print(model_ans_idx, ans_idx)
                                    print("===============================================")

                            elif 'hop1' in version:
                                for c, a1, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
                                                                         model_ans_idxs, ans_idxs):
                                    for idx in (-a1).argsort()[:10]:
                                        prob = "{0:2f}".format(a1[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                                    print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                                    print(model_ans_idx, ans_idx)
                                    print("===============================================")

                            print_only_first = False

                        acc, loss_val = sess.run([accuracy, loss], feed_dict=feed_dict)

                        accs.append(acc)
                        losses.append(loss_val)
                        val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                        summary_writer.add_summary(val_summary, current_step)
                    mean_acc = sum(accs) / len(accs)
                    mean_loss = sum(losses) / len(losses)
                    log.warning("val_loss: {} / val_acc: {}".format(mean_loss, mean_acc))
                    try:
                        if mean_acc > max_acc:
                            alarm.send_message(
                                "babi-dialog {4} {3} {2} val loss:{1:.4f} val acc:{0:.4f}".format(
                                    mean_acc, mean_loss, version, current_step, task_idx),
                                channel='general')
                            max_acc = mean_acc
                            saver.save(sess, save_path=save_summary_path, global_step=current_step)
                        elif (mean_acc == max_acc) and (mean_loss < min_loss):
                            alarm.send_message(
                                "babi-dialog {4} {3} {2} val loss:{1:.4f} val acc:{0:.4f}".format(
                                    mean_acc, mean_loss, version, current_step, task_idx),
                                channel='general')
                            min_loss = mean_loss
                            saver.save(sess, save_path=save_summary_path, global_step=current_step)
                    except:
                        print("alarm error")

                    print("====training====")
            end_time = time()
            alarm.send_message("End training babi-dialog {}".format(version), channel='general')
            print("Training finished in {}sec".format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metagraph', '--metagraph')
    parser.add_argument('--task', type=int)
    parser.add_argument('--is_oov', type=bool, default=False)
    parser.add_argument('--gpu_frac', type=float, default=0.45)
    parser.add_argument('--display_step', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iter_time', type=int, default=1000)
    parser.add_argument('--use_test', type=bool, default=False)
    config = parser.parse_args()

    run(config)
