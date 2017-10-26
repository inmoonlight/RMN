import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.python import debug as tf_debug
from time import time

sys.path.append('../../')

import alarm
from model import Model
from input_ops import load_data, batch_iter, parse_config_txt
from util import log

alarm_channel = 'babi_dialog'


def run(config):
    task_idx = config.task
    is_oov = config.is_oov
    use_match = config.use_match
    version = config.version
    gpu_frac = config.gpu_frac
    display_step = config.display_step
    batch_size = config.batch_size
    iter_time = config.iter_time
    learning_rate = config.learning_rate

    if is_oov:
        task_idx = 'oov_{}'.format(task_idx)

    train_dataset = load_data(task=task_idx, type='train')
    if config.use_test:  # TODO: remove when upload to the public github
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
        alarm.send_message("Start running Model-v{}-task_{} ...".format(version, task_idx), channel=alarm_channel)
        start_time = time()
        with sess.as_default():
            tf.set_random_seed(config.seed)
            m = Model(config)
            pred, correct, accuracy, loss, sim_score, p, a = getattr(m, 'v{}'.format(version))()

            # create model directory
            date = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
            model_id = "Model-v{}-g_theta-{}-f_phi-{}-seed_{}-{}".format(version,
                                                                         '-'.join(
                                                                             str(m.g_theta_layers)[1:-1].split(", ")),
                                                                         '-'.join(
                                                                             str(m.f_phi_layers)[1:-1].split(", ")),
                                                                         config.seed, date)
            if use_match:
                save_dir = "./result/task_{}_{}/{}/".format(task_idx, 'use_match', model_id)
            else:
                save_dir = "./result/task_{}/{}/".format(task_idx, model_id)
            save_summary_path = os.path.join(save_dir, 'model_summary')
            save_variable_path = os.path.join(save_dir, 'model_variables')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.makedirs(save_summary_path)
                os.makedirs(save_variable_path)

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(learning_rate)
            optimizer = opt.minimize(loss, global_step=global_step)
            tf.add_to_collection("optimizer", optimizer)

            loss_train = tf.summary.scalar("loss_train", loss)
            accuracy_train = tf.summary.scalar("accuracy_train", accuracy)
            train_summary_ops = tf.summary.merge([loss_train, accuracy_train])

            loss_val = tf.summary.scalar("loss_val", loss)
            accuracy_val = tf.summary.scalar("accuracy_val", accuracy)
            val_summary_ops = tf.summary.merge([loss_val, accuracy_val])

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(save_summary_path, sess.graph)
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
                feed_dict = {m.context: c_batch,
                             m.context_real_len: c_real_len_batch,
                             m.question: q_batch,
                             m.question_real_len: q_real_len_batch,
                             m.answer: cand_batch,
                             m.answer_match: cand_match_batch,
                             m.answer_idx: a_idx_batch,
                             m.is_training: True}
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
                        model_score, alpha_1, alpha_2, alpha_3, alpha_4, prediction_, answer_ = sess.run(
                            [sim_score, m.alpha_1, m.alpha_2, m.alpha_3, m.alpha_4, p, a], feed_dict=feed_dict)
                    elif 'hop3' in version:
                        model_score, alpha_1, alpha_2, alpha_3, prediction_, answer_ = sess.run(
                            [sim_score, m.alpha_1, m.alpha_2, m.alpha_3, p, a], feed_dict=feed_dict)
                    elif 'hop2' in version:
                        model_score, alpha_1, alpha_2, prediction_, answer_ = sess.run(
                            [sim_score, m.alpha_1, m.alpha_2, p, a], feed_dict=feed_dict)
                    elif 'hop1' in version:
                        model_score, alpha_1, prediction_, answer_ = sess.run(
                            [sim_score, m.alpha_1, p, a], feed_dict=feed_dict)
                    else:
                        model_score, prediction_, answer_ = sess.run([sim_score, p, a], feed_dict=feed_dict)
                    model_ans_idxs = [np.argmax(score, 0) for score in model_score[:3]]
                    ans_idxs = [idx for idx in a_idx_batch[:3]]

                    if 'hop4' in version:
                        for c, a_1, a_2, a_3, a_4, model_ans_idx, ans_idx in zip(c_batch[:1], np.squeeze(alpha_1)[:1],
                                                                                 np.squeeze(alpha_2)[:1],
                                                                                 np.squeeze(alpha_3)[:1],
                                                                                 np.squeeze(alpha_4)[:1],
                                                                                 model_ans_idxs, ans_idxs):
                            for idx in (-a_1).argsort()[:10]:
                                prob = "{0:2f}".format(a_1[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            print("\n")

                            for idx in (-a_2).argsort()[:10]:
                                prob = "{0:2f}".format(a_2[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            print("\n")

                            for idx in (-a_3).argsort()[:10]:
                                prob = "{0:2f}".format(a_3[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            print("\n")

                            for idx in (-a_4).argsort()[:10]:
                                prob = "{0:2f}".format(a_4[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))
                            # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                            # print("pred: {}".format(prediction_[:1][0]))
                            # print("answer: {}".format(answer_[:1][0][ans_idx]))
                            # print("anything: {}".format(answer_[:1][0][109]))
                            print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                            print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                            print(model_ans_idx, ans_idx)
                            print("===============================================")

                    elif 'hop3' in version:
                        for c, a_1, a_2, a_3, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
                                                                            np.squeeze(alpha_2)[:3],
                                                                            np.squeeze(alpha_3)[:3],
                                                                            model_ans_idxs, ans_idxs):
                            for idx in (-a_1).argsort()[:10]:
                                prob = "{0:2f}".format(a_1[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            print("\n")

                            for idx in (-a_2).argsort()[:10]:
                                prob = "{0:2f}".format(a_2[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            print("\n")

                            for idx in (-a_3).argsort()[:10]:
                                prob = "{0:2f}".format(a_3[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                            # print("pred: {}".format(prediction_[:1][0]))
                            # print("answer: {}".format(answer_[:1][0][ans_idx]))
                            # print("anything: {}".format(answer_[:1][0][109]))
                            print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                            print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                            print(model_ans_idx, ans_idx)
                            print("===============================================")

                    elif 'hop2' in version:
                        for c, a_1, a_2, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
                                                                       np.squeeze(alpha_2)[:3],
                                                                       model_ans_idxs, ans_idxs):
                            for idx in (-a_1).argsort()[:10]:
                                prob = "{0:2f}".format(a_1[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            print("\n")

                            for idx in (-a_2).argsort()[:10]:
                                prob = "{0:2f}".format(a_2[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                            # print("pred: {}".format(prediction_[:1][0]))
                            # print("answer: {}".format(answer_[:1][0][ans_idx]))
                            # print("anything: {}".format(answer_[:1][0][109]))
                            print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                            print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                            print(model_ans_idx, ans_idx)
                            print("===============================================")

                    elif 'hop1' in version:
                        for c, a_1, model_ans_idx, ans_idx in zip(c_batch[:3], np.squeeze(alpha_1)[:3],
                                                                  model_ans_idxs, ans_idxs):
                            for idx in (-a_1).argsort()[:10]:
                                prob = "{0:2f}".format(a_1[idx])
                                ss = [prob]
                                for c_word_idx in c[idx]:
                                    if c_word_idx == 0:
                                        ss.append("")
                                    else:
                                        ss.append(idx_to_word[c_word_idx - 1])
                                log.critical(" ".join(ss))

                            # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                            # print("pred: {}".format(prediction_[:1][0]))
                            # print("answer: {}".format(answer_[:1][0][ans_idx]))
                            # print("anything: {}".format(answer_[:1][0][109]))
                            print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                            print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                            print(model_ans_idx, ans_idx)
                            print("===============================================")
                    else:
                        for model_ans_idx, ans_idx in zip(model_ans_idxs, ans_idxs):
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
                        feed_dict = {m.context: c_val,
                                     m.context_real_len: c_real_len_val,
                                     m.question: q_val,
                                     m.question_real_len: q_real_len_val,
                                     m.answer: cand_val,
                                     m.answer_match: cand_match_val,
                                     m.answer_idx: a_idx_val,
                                     m.is_training: False}

                        # print model answer and real answer
                        if print_only_first:
                            if 'hop4' in version:
                                model_score, alpha_1, alpha_2, alpha_3, alpha_4, prediction_, answer_ = sess.run(
                                    [sim_score, m.alpha_1, m.alpha_2, m.alpha_3, m.alpha_4, p, a], feed_dict=feed_dict)
                            elif 'hop3' in version:
                                model_score, alpha_1, alpha_2, alpha_3, prediction_, answer_ = sess.run(
                                    [sim_score, m.alpha_1, m.alpha_2, m.alpha_3, p, a], feed_dict=feed_dict)
                            elif 'hop2' in version:
                                model_score, alpha_1, alpha_2, prediction_, answer_ = sess.run(
                                    [sim_score, m.alpha_1, m.alpha_2, p, a], feed_dict=feed_dict)
                            elif 'hop1' in version:
                                model_score, alpha_1, prediction_, answer_ = sess.run(
                                    [sim_score, m.alpha_1, p, a], feed_dict=feed_dict)
                            else:
                                model_score, prediction_, answer_ = sess.run([sim_score, p, a], feed_dict=feed_dict)
                            model_ans_idxs = [np.argmax(score, 0) for score in model_score[:3]]
                            ans_idxs = [idx for idx in a_idx_val[:3]]
                            if 'hop4' in version:
                                for c, a_1, a_2, a_3, a_4, model_ans_idx, ans_idx in zip(c_val[:1],
                                                                                         np.squeeze(alpha_1)[:1],
                                                                                         np.squeeze(alpha_2)[:1],
                                                                                         np.squeeze(alpha_3)[:1],
                                                                                         np.squeeze(alpha_4)[:1],
                                                                                         model_ans_idxs, ans_idxs):
                                    for idx in (-a_1).argsort()[:10]:
                                        prob = "{0:2f}".format(a_1[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])

                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a_2).argsort()[:10]:
                                        prob = "{0:2f}".format(a_2[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a_3).argsort()[:10]:
                                        prob = "{0:2f}".format(a_3[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a_4).argsort()[:10]:
                                        prob = "{0:2f}".format(a_4[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))
                                    # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                                    # print("pred: {}".format(prediction_[:1][0]))
                                    # print("answer: {}".format(answer_[:1][0][ans_idx]))
                                    # print("anything: {}".format(answer_[:1][0][109]))
                                    print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                                    print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                                    print(model_ans_idx, ans_idx)
                                    print("===============================================")

                            elif 'hop3' in version:
                                for c, a_1, a_2, a_3, model_ans_idx, ans_idx in zip(c_val[:3], np.squeeze(alpha_1)[:3],
                                                                                    np.squeeze(alpha_2)[:3],
                                                                                    np.squeeze(alpha_3)[:3],
                                                                                    model_ans_idxs, ans_idxs):
                                    for idx in (-a_1).argsort()[:10]:
                                        prob = "{0:2f}".format(a_1[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])

                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a_2).argsort()[:10]:
                                        prob = "{0:2f}".format(a_2[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a_3).argsort()[:10]:
                                        prob = "{0:2f}".format(a_3[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))
                                    # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                                    # print("pred: {}".format(prediction_[:1][0]))
                                    # print("answer: {}".format(answer_[:1][0][ans_idx]))
                                    # print("anything: {}".format(answer_[:1][0][109]))
                                    print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                                    print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                                    print(model_ans_idx, ans_idx)
                                    print("===============================================")

                            elif 'hop2' in version:
                                for c, a_1, a_2, model_ans_idx, ans_idx in zip(c_val[:3], np.squeeze(alpha_1)[:3],
                                                                               np.squeeze(alpha_2)[:3],
                                                                               model_ans_idxs, ans_idxs):
                                    for idx in (-a_1).argsort()[:10]:
                                        prob = "{0:2f}".format(a_1[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])

                                        log.error(" ".join(ss))

                                    print("\n")

                                    for idx in (-a_2).argsort()[:10]:
                                        prob = "{0:2f}".format(a_2[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])
                                        log.error(" ".join(ss))

                                    # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                                    # print("pred: {}".format(prediction_[:1][0]))
                                    # print("answer: {}".format(answer_[:1][0][ans_idx]))
                                    # print("anything: {}".format(answer_[:1][0][109]))
                                    print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                                    print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                                    print(model_ans_idx, ans_idx)
                                    print("===============================================")

                            elif 'hop1' in version:
                                for c, a_1, model_ans_idx, ans_idx in zip(c_val[:3], np.squeeze(alpha_1)[:3],
                                                                          model_ans_idxs, ans_idxs):
                                    for idx in (-a_1).argsort()[:10]:
                                        prob = "{0:2f}".format(a_1[idx])
                                        ss = [prob]
                                        for c_word_idx in c[idx]:
                                            if c_word_idx == 0:
                                                ss.append("")
                                            else:
                                                ss.append(idx_to_word[c_word_idx - 1])

                                        log.error(" ".join(ss))

                                    # print("pred_ans: {}".format(answer_[:1][0][model_ans_idx]))
                                    # print("pred: {}".format(prediction_[:1][0]))
                                    # print("answer: {}".format(answer_[:1][0][ans_idx]))
                                    # print("anything: {}".format(answer_[:1][0][109]))
                                    print("Model answer: {}".format(idx_to_cand[model_ans_idx]))
                                    print("Real  answer: {}".format(idx_to_cand[ans_idx]))
                                    print(model_ans_idx, ans_idx)
                                    print("===============================================")
                            else:
                                for model_ans_idx, ans_idx in zip(model_ans_idxs, ans_idxs):
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
                                "babi-dialog task {3} seed{4} g_theta {6} f_phi {7} v.{2} {1} val loss:{5:.4f} val acc:{0:.4f}".format(
                                    mean_acc, current_step, version, task_idx, config.seed, mean_loss, m.g_theta_layers,
                                    m.f_phi_layers),
                                channel='general')
                            max_acc = mean_acc
                            saver.save(sess, save_path=save_summary_path, global_step=current_step)
                        elif (mean_acc == max_acc) and (mean_loss < min_loss):
                            alarm.send_message(
                                "babi-dialog task {3} seed{4} g_theta {6} f_phi {7} v.{2} {1} val loss:{5:.4f} val acc:{0:.4f}".format(
                                    mean_acc, current_step, version, task_idx, config.seed, mean_loss, m.g_theta_layers,
                                    m.f_phi_layers),
                                channel='general')
                            min_loss = mean_loss
                            saver.save(sess, save_path=save_summary_path, global_step=current_step)

                        if task_idx == 4:
                            if (max_acc > 60) and (mean_acc < 40):
                                alarm.send_message('early stop for task 4 g_theta {} f_phi {}'.format(m.g_theta_layers,
                                                                                                      m.f_phi_layers))
                                exit()
                    except:
                        print("alarm error")

                    if config.find:
                        if current_step > config.threshold_step:
                            if max_acc < config.threshold_acc:
                                alarm.send_message('seed {} program exit'.format(config.seed))
                                exit()
                    print("====training====")
            end_time = time()
            alarm.send_message("End training babi-dialog Model-v{}-task{} ...".format(version, task_idx),
                               channel='general')
            if config.find:
                alarm.send_message("the seed {} is good!".format(config.seed))
            print("Training finished in {}sec".format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int)
    parser.add_argument('--is_oov', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--use_match', type=bool, default=False)
    parser.add_argument('--version', '--version', help='model')
    parser.add_argument('--embedding', type=str, help='embedding method of context and question, sum or concat')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--gpu_frac', type=float, default=0.45)
    parser.add_argument('--display_step', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iter_time', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--word_embed_dim', type=int, default=64)  # embedding lookup dim
    parser.add_argument('--hidden_dim', type=int, default=64)  # lstm hidden dim
    parser.add_argument('--use_test', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--find', type=bool, default=False)
    parser.add_argument('--threshold_step', type=int, default=10000)
    parser.add_argument('--threshold_acc', type=float, default=0.4)
    config = parser.parse_args()

    run(config)
