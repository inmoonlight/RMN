import numpy as np
import pandas as pd
import os
import sys
import pickle

sys.path.append('../../')

import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from time import time

import alarm
from model import Model
from utils import read_data, batch_iter, parse_config

dir(tf.contrib)
# flags setting
flags = tf.app.flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '--version')
    parser.add_argument('--gpu_frac', type=float, default=0.95)
    parser.add_argument('--metagraph', '--metagraph')
    parser.add_argument('--display_step', type=int, default=5000)
    parser.add_argument('--task', type=int, default=-1)
    _version = parser.parse_args().version  # 8_1
    _gpu_frac = parser.parse_args().gpu_frac
    _meta = parser.parse_args().metagraph
    _display_step = parser.parse_args().display_step
    _task = parser.parse_args().task

    global config
    with open('config_' + str(_task) + '.txt', 'r') as f:
    # with open('config.txt', 'r') as f:
        config = parse_config(f.readline())

    save_dir = "/".join(_meta.split("/")[:-1])
    save_summary_path = os.path.join(save_dir, 'model_summary')
    save_variable_path = os.path.join(save_dir, 'model_variables')
    checkpoint = '/'.join(_meta.split('/')[:-1])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_summary_path)
        os.makedirs(save_variable_path)

    (train_dataset, val_dataset) = read_data(config['babi_processed'])

    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess_config.gpu_options.per_process_gpu_memory_fraction = _gpu_frac
        sess = tf.Session(config=sess_config)
        max_acc = 0
        try:
            alarm.send_message("Start retraining Model-v{}...".format(_version), channel="babi")
        except:
            pass
        start_time = time()
        with sess.as_default():
            saver = tf.train.import_meta_graph(_meta)
            print("checkpoint : {}".format(tf.train.latest_checkpoint(checkpoint)))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            assert _meta == str(tf.train.latest_checkpoint(checkpoint)) + '.meta'

            graph = tf.get_default_graph()
            # placeholder

            global_step = graph.get_tensor_by_name("global_step:0")

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

            optimizer = graph.get_collection("optimizer")[0]
            correct = graph.get_collection("correct")[0]
            loss = graph.get_collection("loss")[0]
            accuracy = tf.reduce_mean(correct)
            # accuracy = graph.get_collection("accuracy")[0]

            loss_train = tf.summary.scalar("loss_train", loss)
            accuracy_train = tf.summary.scalar("accuracy_train", accuracy)
            train_summary_ops = tf.summary.merge([loss_train, accuracy_train])

            loss_val = tf.summary.scalar("loss_val", loss)
            accuracy_val = tf.summary.scalar("accuracy_val", accuracy)
            val_summary_ops = tf.summary.merge([loss_val, accuracy_val])

            # print([n.name for n in tf.get_default_graph().as_graph_def().node if "mix_with_q/a" in n.name])

            a_1 = graph.get_tensor_by_name("a_1/alpha:0")
            a_2 = graph.get_tensor_by_name("a_2/alpha:0")

            summary_writer = tf.summary.FileWriter(save_summary_path, graph)
            batch_train = batch_iter(c=train_dataset[3],
                                     q=train_dataset[0],
                                     a=train_dataset[1],
                                     a_num=train_dataset[2],
                                     l=train_dataset[4],
                                     c_real_len=train_dataset[5],
                                     q_real_len=train_dataset[6],
                                     h=train_dataset[7],
                                     num_epochs=config['iter_time'],
                                     batch_size=config['batch_size'],
                                     _version=_version,
                                     is_training=True)
            for train in batch_train:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch, h_batch = zip(
                    *train)
                feed_dict = {context: c_batch,
                             question: q_batch,
                             label: l_batch,
                             answer: a_batch,
                             answer_num: a_num_batch,
                             sentence_real_len: c_real_len_batch,
                             question_real_len: q_real_len_batch,
                             hint: h_batch,
                             is_training: True}
                current_step = sess.run(global_step, feed_dict=feed_dict)
                optimizer.run(feed_dict=feed_dict)
                train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
                summary_writer.add_summary(train_summary, current_step)
                if current_step % _display_step == 0:
                    print(np.argwhere(h_batch[0] == 1))
                    max_sent_num = np.argwhere(c_real_len_batch[0] != 0)[-1][0]
                    print(max_sent_num)
                    a_1_idx = (np.argsort(-a_1.eval(feed_dict=feed_dict)[0][0]))[:10]
                    print(np.apply_along_axis(lambda x: np.where(x <= max_sent_num, x, '.'), 0, a_1_idx))
                    print((a_1.eval(feed_dict=feed_dict)[0][0])[a_1_idx])
                    a_2_idx = (np.argsort(-a_2.eval(feed_dict=feed_dict)[0][0]))[:10]
                    print(np.apply_along_axis(lambda x: np.where(x <= max_sent_num, x, '.'), 0, a_2_idx))
                    print((a_2.eval(feed_dict=feed_dict)[0][0])[a_2_idx])

                    print("step: {}".format(current_step))
                    print("====validation start====")
                    batch_val = batch_iter(c=val_dataset[3],
                                           q=val_dataset[0],
                                           a=val_dataset[1],
                                           a_num=val_dataset[2],
                                           l=val_dataset[4],
                                           c_real_len=val_dataset[5],
                                           q_real_len=val_dataset[6],
                                           h=val_dataset[7],
                                           num_epochs=1,
                                           batch_size=config['batch_size'],
                                           _version=_version,
                                           is_training=False)
                    accs = []
                    for val in batch_val:
                        c_val, q_val, l_val, a_val, a_num_val, c_real_len_val, q_real_len_val, h_val = zip(*val)
                        feed_dict = {context: c_val,
                                     question: q_val,
                                     label: l_val,
                                     answer: a_val,
                                     answer_num: a_num_val,
                                     sentence_real_len: c_real_len_val,
                                     question_real_len: q_real_len_val,
                                     hint: h_val,
                                     is_training: False}

                        acc = accuracy.eval(feed_dict=feed_dict)
                        accs.append(acc)
                        val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                        summary_writer.add_summary(val_summary, current_step)
                    mean_acc = sum(accs) / len(accs)
                    print("Mean accuracy=" + str(mean_acc))
                    if mean_acc < 0.10:
                        break
                    try:
                        if mean_acc > max_acc:
                            alarm.send_message("v.{2} {1:5d} val acc:{0:.4f}".format(mean_acc, current_step, _version), channel='general')
                            max_acc = mean_acc
                            saver.save(sess, save_path=save_summary_path, global_step=current_step)
                    except:
                        print("alarm error")
                    print("====training====")
        end_time = time()
        alarm.send_message("End training Model-v-{}...".format(_version), channel='general')
        print("Training finished in {}sec".format(end_time - start_time))


if __name__ == '__main__':
    main()
