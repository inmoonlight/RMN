import argparse
from datetime import datetime
import numpy as np
import os

import tensorflow as tf

from model import RMN
from utils import read_data, batch_iter, parse_config


def run(config):
    display_step = config.display_step
    seed = config.seed
    iter_time = config.iter_time
    word_embed_dim = config.word_embed_dim
    hidden_dim = config.hidden_dim
    batch_size = config.batch_size
    lr = config.learning_rate

    with open('babi_story/config.txt', 'r') as f:
        config_txt = parse_config(f.readline())

    save_dir = "./result/babi_story/"
    save_summary_path = os.path.join(save_dir, 'model_summary')
    save_variable_path = os.path.join(save_dir, 'model_variables')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_summary_path)
        os.makedirs(save_variable_path)

    (train_dataset, val_dataset) = read_data(config_txt['babi_processed'])

    with tf.Graph().as_default():
        sess = tf.Session()
        max_acc = 0
        with sess.as_default():
            tf.set_random_seed(seed)
            m = RMN(config, config_txt, seed=seed, word_embed_dim=word_embed_dim, hidden_dim=hidden_dim)
            pred, correct, accuracy, loss = m.run()

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
            batch_train = batch_iter(c=train_dataset[3],
                                     q=train_dataset[0],
                                     l=train_dataset[4],
                                     a=train_dataset[1],
                                     a_num=train_dataset[2],
                                     c_real_len=train_dataset[5],
                                     q_real_len=train_dataset[6],
                                     batch_size=batch_size,
                                     num_epochs=iter_time,
                                     shuffle=True)
            for train in batch_train:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch = zip(*train)
                feed_dict = {m.context: c_batch,
                             m.question: q_batch,
                             m.label: l_batch,
                             m.answer: a_batch,
                             m.answer_num: a_num_batch,
                             m.sentence_real_len: c_real_len_batch,
                             m.question_real_len: q_real_len_batch,
                             m.is_training: True}
                current_step = sess.run(global_step, feed_dict=feed_dict)
                optimizer.run(feed_dict=feed_dict)
                train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
                summary_writer.add_summary(train_summary, current_step)
                if current_step % display_step == 0:
                    print("step: {}".format(current_step))
                    print("====validation start====")
                    batch_val = batch_iter(c=val_dataset[3],
                                           q=val_dataset[0],
                                           l=val_dataset[4],
                                           a=val_dataset[1],
                                           a_num=val_dataset[2],
                                           c_real_len=val_dataset[5],
                                           q_real_len=val_dataset[6],
                                           num_epochs=1,
                                           batch_size=batch_size,
                                           shuffle=False)
                    accs = []
                    for val in batch_val:
                        c_val, q_val, l_val, a_val, a_num_val, c_real_len_val, q_real_len_val = zip(*val)
                        feed_dict = {m.context: c_val,
                                     m.question: q_val,
                                     m.label: l_val,
                                     m.answer: a_val,
                                     m.answer_num: a_num_val,
                                     m.sentence_real_len: c_real_len_val,
                                     m.question_real_len: q_real_len_val,
                                     m.is_training: False}
                        acc = accuracy.eval(feed_dict=feed_dict)
                        accs.append(acc)
                        val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                        summary_writer.add_summary(val_summary, current_step)
                    mean_acc = sum(accs) / len(accs)
                    print("Mean accuracy=" + str(mean_acc))
                    if mean_acc > max_acc:
                        max_acc = mean_acc
                        saver.save(sess, save_path=save_summary_path, global_step=current_step)
                    print("====training====")
        print("Training completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_step', type=int, default=5000, help='validation step')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--iter_time', type=int, default=1000, help='training iteration')
    parser.add_argument('--word_embed_dim', type=int, default=32, help='one-hot word vector to d-dimensional vector')
    parser.add_argument('--hidden_dim', type=int, default=32, help='lstm hidden dimension')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    config = parser.parse_args()
    run(config)
