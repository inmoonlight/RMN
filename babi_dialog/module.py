import itertools
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm, fully_connected, layer_norm
from tensorflow.contrib.layers import xavier_initializer

from input_ops import parse_config_txt


class Module:
    def __init__(self, config, g_theta_layers, f_phi_layers):
        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.batch_size = config.batch_size
        self.seed = config.seed
        config_txt = parse_config_txt('config_{}.txt'.format(config.task))
        self.c_max_len = config_txt['memory_size']
        self.s_max_len = config_txt['sentence_size']
        self.q_max_len = config_txt['sentence_size']
        self.a_max_len = config_txt['cand_sentence_size']
        self.vocab_size = config_txt['vocab_size']
        self.cand_size = config_txt['cand_size']
        self._type_num = config_txt['_TYPE_NUM']
        self.mask_index = 0
        self.word_embed_dim = config.word_embed_dim
        self.hidden_dim = config.hidden_dim
        self.word_embed_matrix = self.embed_matrix()
        self.encoding = tf.constant(self.position_encoding(self.s_max_len, self.word_embed_dim), name='encoding')

    def embed_matrix(self):
        word_embed_matrix = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.word_embed_dim], minval=-1,
                                                          maxval=1, seed=self.seed), name='word_embed_matrix')
        return word_embed_matrix

    def bn_module(self, input_, layers, phase, activation=tf.nn.relu):
        outputs = [input_]
        for layer_dim in layers:
            fc_output = fully_connected(outputs[-1], layer_dim, activation_fn=None)
            bn_output = batch_norm(fc_output, decay=0.95, center=True, scale=True, is_training=phase,
                                   updates_collections=None, activation_fn=activation)
            outputs.append(bn_output)
        return outputs[-1]

    def fc_module(self, input_, layers, phase, activation=tf.nn.relu):
        outputs = [input_]
        for layer_dim in layers:
            fc_output = fully_connected(outputs[-1], layer_dim, activation_fn=activation)
            outputs.append(fc_output)
        return outputs[-1]

    def position_encoding(self, sentence_size, embedding_size):
        """
        Position Encoding described in section MemNet 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        # Make position encoding of time words identity to avoid modifying them
        encoding[:, -1] = 1.0
        return np.transpose(encoding)

    def contextSum(self, c, scope='contextSum', with_embed_matrix=True, with_position_encoding=False):
        sentences = tf.reshape(c, shape=[-1, self.s_max_len])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_s_word = tf.nn.embedding_lookup(self.word_embed_matrix, sentences)
            else:
                embedded_s_word = tf.one_hot(indices=sentences, depth=self.vocab_size)

            if with_position_encoding:
                s_sum = tf.reduce_sum(embedded_s_word * self.encoding, axis=1)
            else:
                s_sum = tf.reduce_sum(embedded_s_word, axis=1)

            c_embedded = tf.reshape(s_sum, shape=[self.batch_size, self.c_max_len, -1])
        return tf.unstack(c_embedded, axis=1)

    def contextConcat(self, c, scope='contextConcat', with_embed_matrix=True, with_position_encoding=False):
        sentences = tf.reshape(c, shape=[-1, self.s_max_len])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_s_word = tf.nn.embedding_lookup(self.word_embed_matrix, sentences)
            else:
                embedded_s_word = tf.one_hot(indices=sentences, depth=self.vocab_size)

            if with_position_encoding:
                s_concat = tf.concat(tf.unstack(embedded_s_word * self.encoding, axis=1), axis=1)
            else:
                s_concat = tf.concat(tf.unstack(embedded_s_word, axis=1), axis=1)

            c_embedded = tf.reshape(s_concat, shape=[self.batch_size, self.c_max_len, -1])
        return tf.unstack(c_embedded, axis=1)

    def questionSum(self, q, scope='questionSum', with_embed_matrix=True, with_position_encoding=False):
        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            else:
                embedded_q_word = tf.one_hot(indices=q, depth=self.vocab_size)

            if with_position_encoding:
                q_sum = tf.reduce_sum(embedded_q_word * self.encoding, axis=1)
            else:
                q_sum = tf.reduce_sum(embedded_q_word, axis=1)
        return q_sum

    def questionConcat(self, q, scope='questionConcat', with_embed_matrix=True, with_position_encoding=False):
        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            else:
                embedded_q_word = tf.one_hot(indices=q, depth=self.vocab_size)

            if with_position_encoding:
                q_concat = tf.concat(tf.unstack(embedded_q_word * self.encoding, axis=1), axis=1)
            else:
                q_concat = tf.concat(tf.unstack(embedded_q_word, axis=1), axis=1)
        return q_concat

    def g_theta(self, z, phase=True, activation=tf.nn.tanh, reuse=True, scope=""):
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

    def attention(self, prev_alpha, r, embedded_c, phase=True, activation=tf.nn.tanh, reuse=True, scope=""):
        """
        calculate attention with g_theta

        Args:
            r (decoder output, [batch_size, word_embed])
            embedded_c ([batch_size, c_max_len, word_embed])
        """
        r = tf.tile(tf.expand_dims(r, axis=1), tf.constant([1, self.c_max_len, 1]))
        embedded_c = (1 - tf.transpose(prev_alpha, [0, 2, 1])) * embedded_c
        z = tf.concat([r, embedded_c], axis=2)
        alpha = self.g_theta(z, phase=phase, activation=activation, reuse=reuse, scope=scope)
        c = tf.squeeze(tf.matmul(alpha, embedded_c))  # [batch_size, word_embed]
        return alpha, c

    def hop_1(self, embedded_c, embedded_q, phase=True, activation=tf.nn.tanh):
        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, r_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, activation=activation, reuse=False,
                                      scope="a_1")
        r = [r_1]
        alphas = [alpha_1]
        return r, alphas

    def concat_with_q(self, last_ss, embedded_q):
        return tf.concat([last_ss, embedded_q], axis=1)  # [batch_size, word_embed*2]

    def f_phi(self, g, activation=tf.nn.relu, scope="f_phi", reuse=True, with_embed_matrix=True, is_concat=True,
              use_match=False, phase=True):
        f_units = self.f_phi_layers
        with tf.variable_scope(scope, reuse=reuse) as scope:
            f_output = self.bn_module(g, f_units, phase=phase, activation=activation)
            with tf.variable_scope("pred", reuse=reuse):
                if with_embed_matrix:
                    if is_concat is True and use_match is True:
                        pred = self.fc_module(f_output, [self.word_embed_dim * self.a_max_len + self._type_num],
                                              activation=None,
                                              phase=phase)
                    elif is_concat is True and use_match is False:
                        pred = self.fc_module(f_output, [self.word_embed_dim * self.a_max_len], activation=None,
                                              phase=phase)
                    elif is_concat is False and use_match is True:
                        pred = self.fc_module(f_output, [self.word_embed_dim + self._type_num], activation=None,
                                              phase=phase)
                    elif is_concat is False and use_match is False:
                        pred = self.fc_module(f_output, [self.word_embed_dim], activation=None, phase=phase)
                else:
                    pred = self.fc_module(f_output, [self.vocab_size], activation=None, phase=phase)
        return pred

    def answerSum(self, a, scope='answerSum', with_embed_matrix=True, with_position_encoding=False):
        """
        Args
            a: zero padded answers, shape=[batch_size, answer_size, a_max_len]
            a_real_lens: original answer length, shape = [batch_size, answer_size, 1]

        Returns
            embedded_a: embedded answer, shape = [batch_size, answer_size, hidden(32)]
        """
        answers = tf.reshape(a, shape=[-1, self.a_max_len])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_a_word = tf.nn.embedding_lookup(self.word_embed_matrix,
                                                         answers)  # [batch_size*answer_size, a_max_len, embedding_dim]
            else:
                embedded_a_word = tf.one_hot(indices=answers, depth=self.vocab_size)

            if with_position_encoding:
                encoding = tf.constant(self.position_encoding(self.a_max_len, self.word_embed_dim), name='encoding')
                a_sum = tf.reduce_sum(embedded_a_word * encoding, axis=1)  # [batch_size*answer_size, embedding_dim]
            else:
                a_sum = tf.reduce_sum(embedded_a_word, axis=1)  # [batch_size*answer_size, embedding_dim]

        return tf.reshape(a_sum, shape=[self.batch_size, self.cand_size, -1])

    def answerConcat(self, a, scope='answerConcat', with_embed_matrix=True, with_position_encoding=False):
        """
        Args
            a: zero padded answers, shape=[batch_size, answer_size, a_max_len]
            a_real_lens: original answer length, shape = [batch_size, answer_size, 1]

        Returns
            embedded_a: embedded answer, shape = [batch_size, answer_size, hidden(32)]
        """
        answers = tf.reshape(a, shape=[-1, self.a_max_len])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_a_word = tf.nn.embedding_lookup(self.word_embed_matrix,
                                                         answers)  # [batch_size*answer_size, a_max_len, embedding_dim]
            else:
                embedded_a_word = tf.one_hot(indices=answers, depth=self.vocab_size)

            if with_position_encoding:
                encoding = tf.constant(self.position_encoding(self.a_max_len, self.word_embed_dim), name='encoding')
                a_concat = tf.concat(tf.unstack(embedded_a_word * encoding, axis=1),
                                     axis=1)  # [batch_size*answer_size, embedding_dim]
            else:
                a_concat = tf.concat(tf.unstack(embedded_a_word, axis=1),
                                     axis=1)  # [batch_size*answer_size, embedding_dim]

        return tf.reshape(a_concat, shape=[self.batch_size, self.cand_size, -1])

    def get_corr_acc_loss(self, prediction, a, a_match, answer_idx, with_position_encoding=False,
                          with_embed_matrix=True, is_concat=True, use_match=False, is_cosine_sim=True):
        tf.add_to_collection("prediction", prediction)  # [batch_size, hidden_dim]
        prediction = tf.expand_dims(prediction, axis=1)
        if is_concat:
            embedded_a = self.answerConcat(a, with_position_encoding=with_position_encoding)
        else:
            embedded_a = self.answerSum(a, with_position_encoding=with_position_encoding,
                                        with_embed_matrix=with_embed_matrix)

        prediction = tf.nn.l2_normalize(prediction, dim=2)  # [batch_size, 1, hidden_dim]
        embedded_a = tf.nn.l2_normalize(embedded_a, dim=2)  # [batch_size, answer_size, hidden_dim]

        if use_match:
            embedded_a = tf.concat([embedded_a, a_match], axis=2)

        if is_cosine_sim:
            sim_score = tf.squeeze(tf.matmul(prediction, embedded_a, transpose_b=True))  # [batch_size, answer_size]
        else:
            sim_score = -tf.reduce_sum(tf.abs(tf.subtract(prediction, embedded_a), name='abs_diff'),
                                       axis=-1)  # [batch_size, answer_size]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sim_score, labels=answer_idx))
        max_score_idx = tf.argmax(sim_score, 1)
        tf.add_to_collection("max_score_idx", max_score_idx)
        correct = tf.equal(max_score_idx, answer_idx)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.add_to_collection("loss", loss)
        tf.add_to_collection("correct", correct)
        tf.add_to_collection("accuracy", acc)
        return correct, acc, loss, sim_score, prediction, embedded_a
