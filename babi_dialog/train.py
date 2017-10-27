import argparse
from datetime import datetime
import numpy as np
import os

import tensorflow as tf

from model import RMN
from input_ops import load_data, batch_iter, parse_config_txt
from util import log


def run(config):
    task_idx = config.task
    is_oov = config.is_oov
    use_match = config.use_match
    display_step = config.display_step
    batch_size = config.batch_size
    iter_time = config.iter_time
    lr = config.learning_rate

    if is_oov:
        task_idx = 'oov_{}'.format(task_idx)

    train_dataset = load_data(task=task_idx, type='train')
    val_dataset = load_data(task=task_idx, type='val')
    candidate = load_data(task=task_idx, type='candidate')

    with tf.Graph().as_default():
        sess = tf.Session()
        max_acc = 0
        with sess.as_default():
            tf.set_random_seed(config.seed)
            m = RMN(config)
            pred, correct, accuracy, loss, sim_score, p, a = m.run()

            # create model directory
            if use_match:
                save_dir = "./result/babi_dialog/task_{}_{}/".format(task_idx, 'use_match')
            else:
                save_dir = "./result/babi_dialog/task_{}/".format(task_idx)
            save_summary_path = os.path.join(save_dir, 'model_summary')
            save_variable_path = os.path.join(save_dir, 'model_variables')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.makedirs(save_summary_path)
                os.makedirs(save_variable_path)

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(lr)
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
            batch_train = batch_iter(c=train_dataset[0],
                                     c_real_len=train_dataset[1],
                                     q=train_dataset[2],
                                     q_real_len=train_dataset[3],
                                     cand=candidate,
                                     match_words=train_dataset[4],
                                     a_idx=train_dataset[5],
                                     shuffle=True,
                                     batch_size=batch_size,
                                     num_epochs=iter_time)
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
                    print("==validation==")
                    batch_val = batch_iter(c=val_dataset[0],
                                           c_real_len=val_dataset[1],
                                           q=val_dataset[2],
                                           q_real_len=val_dataset[3],
                                           cand=candidate,
                                           match_words=val_dataset[4],
                                           a_idx=val_dataset[5],
                                           shuffle=True,
                                           batch_size=batch_size,
                                           num_epochs=1)
                    accs = []
                    losses = []
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

                        acc, loss_val = sess.run([accuracy, loss], feed_dict=feed_dict)

                        accs.append(acc)
                        losses.append(loss_val)
                        val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                        summary_writer.add_summary(val_summary, current_step)
                    mean_acc = sum(accs) / len(accs)
                    mean_loss = sum(losses) / len(losses)
                    log.warning("val_loss: {} / val_acc: {}".format(mean_loss, mean_acc))
                    if mean_acc > max_acc:
                        max_acc = mean_acc
                        saver.save(sess, save_path=save_summary_path, global_step=current_step)
                    elif (mean_acc == max_acc) and (mean_loss < min_loss):
                        min_loss = mean_loss
                        saver.save(sess, save_path=save_summary_path, global_step=current_step)

                    print("====training====")
            print("Training Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int)
    parser.add_argument('--is_oov', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--use_match', type=bool, default=False)
    parser.add_argument('--embedding', type=str, help='embedding method of context and question, sum or concat')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--display_step', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iter_time', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--word_embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    config = parser.parse_args()

    run(config)
