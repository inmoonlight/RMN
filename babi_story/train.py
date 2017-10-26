import numpy as np
import os
import sys

sys.path.append('../../')

import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from time import time

import alarm
from model import Model
from utils import read_data, batch_iter, parse_config

# flags setting
flags = tf.app.flags
alarm_channel = 'babi'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '--version')
    parser.add_argument('--gpu_frac', type=float, default=0.95)
    parser.add_argument('--display_step', type=int, default=300)
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--iter_time', type =int, default = 500)
    _version = parser.parse_args().version  # 8_1
    _gpu_frac = parser.parse_args().gpu_frac
    _display_step = parser.parse_args().display_step
    _task = parser.parse_args().task
    _seed = parser.parse_args().seed
    _iter_time = parser.parse_args().iter_time

    global config
    with open('config_' + str(_task) + '.txt', 'r') as f:
    # with open('config.txt', 'r') as f:
        config = parse_config(f.readline())
    date = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
    model_id = "Model-v{0}-{1}-{2}/".format(_version, _task, _seed) + date

    save_dir = "./babi_result/" + model_id
    save_summary_path = os.path.join(save_dir, 'model_summary')
    save_variable_path = os.path.join(save_dir, 'model_variables')

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
#        alarm.send_message("Start running Model-v{}...".format(_version), channel=alarm_channel)
        start_time = time()
        with sess.as_default():
            tf.set_random_seed(_seed)
            m = Model(config, seed=_seed)
            pred, correct, accuracy, loss = getattr(m, 'v{}'.format(_version))()

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(config['learning_rate'])
            optimizer = opt.minimize(loss, global_step=global_step)
            tf.add_to_collection("optimizer", optimizer)

            loss_train = tf.summary.scalar("loss_train", loss)
            accuracy_train = tf.summary.scalar("accuracy_train", accuracy)
            train_summary_ops = tf.summary.merge([loss_train, accuracy_train])

            loss_val = tf.summary.scalar("loss_val", loss)
            accuracy_val = tf.summary.scalar("accuracy_val", accuracy)
            val_summary_ops = tf.summary.merge([loss_val, accuracy_val])
  
            tf.add_to_collection("alpha_1", m.alpha_1)
            tf.add_to_collection("alpha_2", m.alpha_2)

            if _version == 'final_hop3':
                tf.add_to_collection("alpha_3", m.alpha_3)
 
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            sess.run(tf.global_variables_initializer())
            
            # debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            summary_writer = tf.summary.FileWriter(save_summary_path, sess.graph)
            batch_train = batch_iter(c=train_dataset[3],
                                     q=train_dataset[0],
                                     a=train_dataset[1],
                                     a_num=train_dataset[2],
                                     l=train_dataset[4],
                                     c_real_len=train_dataset[5],
                                     q_real_len=train_dataset[6],
                                     h=train_dataset[7],
                                     num_epochs=_iter_time,
                                     batch_size=config['batch_size'],
                                     c_max_len=config['c_max_len'],
                                     _version=_version,
                                     is_training=True)
            for train in batch_train:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch, h_batch = zip(
                    *train)
                feed_dict = {m.context: c_batch,
                             m.question: q_batch,
                             m.label: l_batch,
                             m.answer: a_batch,
                             m.answer_num: a_num_batch,
                             m.sentence_real_len: c_real_len_batch,
                             m.question_real_len: q_real_len_batch,
                             m.hint: h_batch,
                             m.is_training: True}
                current_step = sess.run(global_step, feed_dict=feed_dict)
                optimizer.run(feed_dict=feed_dict)
                train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
                summary_writer.add_summary(train_summary, current_step)
                if current_step % _display_step == 0:
                    if 'RN' not in _version: 
                        print(np.argwhere(h_batch[0] == 1))
                        max_sent_num = np.argwhere(c_real_len_batch[0] != 0)[-1][0]
                        print(max_sent_num)
                        a_1_idx = (np.argsort(-m.alpha_1.eval(feed_dict=feed_dict)[0][0]))[:10]
                        print(np.apply_along_axis(lambda x: np.where(x <= max_sent_num, x, '.'), 0, a_1_idx))
                        print((m.alpha_1.eval(feed_dict=feed_dict)[0][0])[a_1_idx])
                        a_2_idx = (np.argsort(-m.alpha_2.eval(feed_dict=feed_dict)[0][0]))[:10]
                        print(np.apply_along_axis(lambda x: np.where(x <= max_sent_num, x, '.'), 0, a_2_idx))
                        print((m.alpha_2.eval(feed_dict=feed_dict)[0][0])[a_2_idx])
                        if _version == 'final_hop3':
                            a_3_idx = (np.argsort(-m.alpha_3.eval(feed_dict=feed_dict)[0][0]))[:10]
                            print(np.apply_along_axis(lambda x: np.where(x <= max_sent_num, x, '.'), 0, a_3_idx))
                            print((m.alpha_3.eval(feed_dict=feed_dict)[0][0])[a_3_idx])

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
                                           c_max_len=config['c_max_len'],
                                           batch_size=config['batch_size'],
                                           _version=_version,
                                           is_training=False)
                    accs = []
                    for val in batch_val:
                        c_val, q_val, l_val, a_val, a_num_val, c_real_len_val, q_real_len_val, h_val = zip(*val)
                        feed_dict = {m.context: c_val,
                                     m.question: q_val,
                                     m.label: l_val,
                                     m.answer: a_val,
                                     m.answer_num: a_num_val,
                                     m.sentence_real_len: c_real_len_val,
                                     m.question_real_len: q_real_len_val,
                                     m.hint: h_val,
                                     m.is_training: False}

                        acc = accuracy.eval(feed_dict=feed_dict)
                        accs.append(acc)
                        val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                        summary_writer.add_summary(val_summary, current_step)
                    mean_acc = sum(accs) / len(accs)
                    print("Mean accuracy=" + str(mean_acc))
                    try:
                        if mean_acc > max_acc:
                            alarm.send_message(
                                "v.{2} task {3} g_theta_layers {4} f_phi_layers {5} seed {6} {1:5d} val acc:{0:.4f}".format(
                                    mean_acc, current_step, _version, _task, m.g_theta_layers, m.f_phi_layers, _seed), channel='task-wise')
                            max_acc = mean_acc
                            saver.save(sess, save_path=save_summary_path, global_step=current_step)
                    except:
                        print("alarm error")
#                     if mean_acc > 0.98:
# #                        alarm.send_message("v.{2} tasl {3} g_theta_layers{4} f_phi_layers{5} seed {6} {1:5d} val acc:{0:.4f}".format(mean_acc, current_step, _version, _task, m.g_theta_layers, m.f_phi_layers, _seed), channel ='general')
# #                        alarm.send_message("task {0} seed {1} step {3}  acc{2} finished!!".format(_task, _seed, mean_acc, current_step), channel = 'task{}'.format(_task))
#                         exit()
#                     if current_step >=20000 and max_acc < 0.85:
#                         alarm.send_message("restart", channel = 'task{}'.format(_task))
#                         exit()
                    print("====training====")
        end_time = time()
        alarm.send_message("End training Model-v{0}... task {2} seed {3} with {1}".format(_version, max_acc, _task, _seed), channel='task{}'.format(_task))
        print("Training finished in {}sec".format(end_time - start_time))


if __name__ == '__main__':
    main()
