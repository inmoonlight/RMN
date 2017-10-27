import itertools
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected


class Module():
    def __init__(self, config_txt, g_theta_layers, f_phi_layers, seed=9, word_embed=32, hidden_dim=32):
        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.batch_size = config_txt['batch_size']
        self.seed = seed
        self.c_max_len = config_txt['c_max_len']  # 130
        self.s_max_len = config_txt['s_max_len']  # 12
        self.q_max_len = config_txt['q_max_len']  # 12
        self.mask_index = 0
        self.word_embed = word_embed
        self.hidden_dim = hidden_dim
        self.answer_vocab_size = config_txt['vocab_size']  # 159s
        self.word_embed_matrix = self.embed_matrix()

    def embed_matrix(self):
        word_embed_matrix = tf.Variable(
            tf.random_uniform(shape=[self.answer_vocab_size + 1, self.word_embed], minval=-1, maxval=1, seed=self.seed))
        return word_embed_matrix

    def bn_module(self, input_, layers, phase, activation=tf.nn.relu):
        outputs = [input_]
        for layer_dim in layers:
            fc_output = fully_connected(outputs[-1], layer_dim, activation_fn=None)
            bn_output = batch_norm(fc_output, decay=0.95, center=True, scale=True, is_training=phase,
                                   updates_collections=None, activation_fn=activation)
            outputs.append(bn_output)
        return outputs[-1]

    def contextLSTM(self, c, s_real_lens, reuse=False, scope="contextLSTM"):
        def sentenceLSTM(s,
                         s_real_len,
                         reuse=reuse,
                         scope="sentenceLSTM"):
            """
            embedding sentence

            Args:
                s: sentence (word index list), shape = [batch_size*130, 12]
                s_real_len: length of the sentence before zero padding, int32

            Returns:
                embedded_s: embedded sentence, shape = [batch_size*130, hidden_dim]
            """
            with tf.variable_scope(scope):
                embedded_sentence_word = tf.nn.embedding_lookup(self.word_embed_matrix, s)
                s_input = tf.unstack(embedded_sentence_word, num=self.s_max_len, axis=1)
                lstm_cell = rnn.BasicLSTMCell(self.hidden_dim, reuse=reuse)
                outputs, _ = rnn.static_rnn(lstm_cell, s_input, dtype=tf.float32)
                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                index = tf.range(0, self.batch_size * self.c_max_len) * self.s_max_len + (s_real_len - 1)
                outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
            return outputs

        """
        contextLSTM without label
        
        Args:
            c: list of sentences, shape = [batch_size, 130, 12]
            s_real_lens: list of real length, shape = [batch_size, 130]

        Returns:
            c_embedded: list of embedded sentence, shape = [batch_size, hidden_dim]
            len(c_embedded) = 130
        """
        with tf.variable_scope(scope):
            sentences = tf.reshape(c, shape=[-1, self.s_max_len])
            real_lens = tf.reshape(s_real_lens, shape=[-1])
            s_embedded = sentenceLSTM(sentences, real_lens, reuse=reuse)
            c_embedded = tf.reshape(s_embedded, shape=[self.batch_size, self.c_max_len, self.hidden_dim])
            c_embedded = tf.unstack(c_embedded, axis=1)
        return c_embedded

    def questionLSTM(self, q, q_real_len, reuse=False, scope="questionLSTM"):
        """
        Args
            q: zero padded questions, shape=[batch_size, q_max_len]
            q_real_len: original question length, shape = [batch_size, 1]

        Returns
            embedded_q: embedded questions, shape = [batch_size, hidden_dim]
        """
        with tf.variable_scope(scope):
            embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            q_input = tf.unstack(embedded_q_word, num=self.q_max_len, axis=1)
            lstm_cell = rnn.BasicLSTMCell(self.hidden_dim, reuse=reuse)
            outputs, _ = rnn.static_rnn(lstm_cell, q_input, dtype=tf.float32)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            index = tf.range(0, self.batch_size) * self.q_max_len + (q_real_len - 1)
            embedded_q = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
        return embedded_q

    def add_label(self, embedded_c, l, scope='add_label'):
        """
        Args:
            embedded_c: list of embedded sentence, shape = [batch_size, hidden_dim]
            l: labels, shape = [batch_size, 130]
            embedded_q: embedded questions, shape = [batch_size, hidden_dim]

        Returns:
            concat_result: shape = [batch_size, hidden_dim + 1]
        """
        with tf.variable_scope(scope):
            embedded_c = tf.stack(embedded_c, axis=1)
            labels = tf.expand_dims(l, axis=2)  # [batch_size, 130, 1]
            concated = tf.concat([embedded_c, labels], axis=2)
        return tf.unstack(concated, axis=1)

    def g_theta(self, z, phase=True, activation=tf.nn.relu, reuse=True, scope=""):
        z = tf.reshape(z, shape=[self.batch_size * self.c_max_len, -1])
        g_units = self.g_theta_layers
        assert g_units[-1] == 1  # attention should be ended with layer sized 1
        with tf.variable_scope(scope, reuse=reuse):
            w = self.bn_module(z, g_units, phase=phase, activation=activation)
            w = tf.reshape(w, shape=[self.batch_size, g_units[-1], self.c_max_len])
            init = tf.contrib.layers.xavier_initializer(seed=self.seed)
            strength = tf.tile(
                1 + tf.nn.softplus(tf.get_variable('strength', shape=[1, 1, 1], initializer=init)),
                tf.constant([self.batch_size, 1, self.c_max_len]))
            alpha = tf.nn.softmax(strength * w, name='alpha')  # [batch_size, 1, c_max_len]
        return alpha

    def attention(self, prev_alpha, r, embedded_c, phase=True, activation=tf.nn.relu, reuse=True, scope=""):
        """
        calculate attention with g_theta

        Args:
            w (decoder output, [batch_size, word_embed])
            c ([batch_size, c_max_len, word_embed])
        """
        r = tf.tile(tf.expand_dims(r, axis=1), tf.constant([1, self.c_max_len, 1]))
        embedded_c = (1 - tf.transpose(prev_alpha, [0, 2, 1])) * embedded_c
        z = tf.concat([r, embedded_c], axis=2)
        alpha = self.g_theta(z, phase=phase, activation=activation, reuse=reuse, scope=scope)
        c = tf.squeeze(tf.matmul(alpha, embedded_c))
        return alpha, c

    def hop_2(self, embedded_c, embedded_q, phase=True, activation=tf.nn.relu):
        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)
        alpha_1, r_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, activation=activation,
                                      reuse=False, scope="a_1")
        alpha_2, r_2 = self.attention(alpha_1, r_1, embedded_c, phase=phase, activation=activation,
                                      reuse=False, scope="a_2")
        r = [r_1, r_2]
        alphas = [alpha_1, alpha_2]
        return r, alphas

    def concat_with_q(self, last_ss, embedded_q):
        return tf.concat([last_ss, embedded_q], axis=1)

    def f_phi(self, g, scope="f_phi", reuse=True, phase=True):
        f_units = self.f_phi_layers
        with tf.variable_scope(scope, reuse=reuse) as scope:
            f_output = self.bn_module(g, f_units, phase=phase, activation=tf.nn.relu)
            with tf.variable_scope("pred_1", reuse=reuse):
                pred_1 = self.bn_module(f_output, [self.answer_vocab_size], activation=None, phase=phase)
            with tf.variable_scope("pred_2", reuse=reuse):
                pred_2 = self.bn_module(f_output, [self.answer_vocab_size], activation=None, phase=phase)
            with tf.variable_scope("pred_3", reuse=reuse):
                pred_3 = self.bn_module(f_output, [self.answer_vocab_size], activation=None, phase=phase)
        pred = tf.concat([pred_1, pred_2, pred_3], axis=1)
        return pred

    def get_corr_acc_loss(self, pred, answer, answer_num):
        answer_bool = tf.split(answer_num, 3, axis=1)
        a_s = tf.unstack(answer, axis=1)
        final_pred = tf.one_hot(tf.argmax(pred[0], axis=1), depth=self.answer_vocab_size) * answer_bool[0] + tf.one_hot(
            tf.argmax(pred[1], axis=1), depth=self.answer_vocab_size) * answer_bool[1] + tf.one_hot(
            tf.argmax(pred[2], axis=1), depth=self.answer_vocab_size) * answer_bool[2]
        final_answer = a_s[0] * answer_bool[0] + a_s[1] * answer_bool[1] + a_s[2] * answer_bool[2]
        tf.add_to_collection("final_pred", final_pred)
        tf.add_to_collection("final_answer", final_answer)
        correct = tf.cast(
            tf.greater_equal(tf.reduce_mean(tf.cast(tf.equal(final_pred, final_answer), tf.int32), axis=1), 1),
            tf.float32)
        tf.add_to_collection("correct", correct)
        accuracy = tf.reduce_mean(correct)
        tf.add_to_collection("accuracy", accuracy)
        loss_before = -tf.reduce_sum(
            a_s[0] * tf.log(tf.nn.softmax(pred[0])) * answer_bool[0] + a_s[1] * tf.log(tf.nn.softmax(pred[1])) *
            answer_bool[1] + a_s[2] * tf.log(tf.nn.softmax(pred[2])) * answer_bool[2], 1)
        tf.add_to_collection('loss_before', loss_before)
        loss = tf.reduce_mean(loss_before)
        tf.add_to_collection('loss', loss)
        return correct, accuracy, loss
