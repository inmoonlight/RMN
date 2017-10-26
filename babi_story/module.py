import itertools
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected


class Module():
    def __init__(self, config, g_theta_layers, f_phi_layers, seed=9, word_embed=32, answer_vocab_size=159):
        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.batch_size = config['batch_size']
        self.seed = seed
        self.c_max_len = config['c_max_len']  # 130
        self.s_max_len = config['s_max_len']  # 12
        self.q_max_len = config['q_max_len']  # 12
        self.mask_index = 0
        self.word_embed = word_embed
        self.answer_vocab_size = answer_vocab_size
        self.word_embed_matrix = self.embed_matrix()

    def embed_matrix(self):
        word_embed_matrix = tf.Variable(
            tf.random_uniform(shape=[self.answer_vocab_size + 1, self.word_embed], minval=-1,
                              maxval=1, seed=self.seed))
        return word_embed_matrix

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
                embedded_s: embedded sentence, shape = [batch_size*130, 32]
            """
            with tf.variable_scope(scope):
                embedded_sentence_word = tf.nn.embedding_lookup(self.word_embed_matrix, s)
                s_input = tf.unstack(embedded_sentence_word, num=self.s_max_len, axis=1)
                lstm_cell = rnn.BasicLSTMCell(self.word_embed, reuse=reuse)
                outputs, _ = rnn.static_rnn(lstm_cell, s_input, dtype=tf.float32)
                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                index = tf.range(0, self.batch_size * self.c_max_len) * self.s_max_len + (s_real_len - 1)
                outputs = tf.gather(tf.reshape(outputs, [-1, self.word_embed]), index)
            return outputs

        """
        contextLSTM without label
        
        Args:
            c: list of sentences, shape = [batch_size, 130, 12]
            s_real_lens: list of real length, shape = [batch_size, 130]

        Returns:
            c_embedded: list of embedded sentence, shape = [batch_size, 32] 130개
            len(c_embedded) = 130
        """
        with tf.variable_scope(scope):
            sentences = tf.reshape(c, shape=[-1, self.s_max_len])
            real_lens = tf.reshape(s_real_lens, shape=[-1])

            s_embedded = sentenceLSTM(sentences, real_lens, reuse=reuse)
            c_embedded = tf.reshape(s_embedded, shape=[self.batch_size, self.c_max_len, self.word_embed])
            c_embedded = tf.unstack(c_embedded, axis=1)
        return c_embedded

    def questionLSTM(self, q, q_real_len, reuse=False, scope="questionLSTM"):
        """
        Args
            q: zero padded questions, shape=[batch_size, q_max_len]
            q_real_len: original question length, shape = [batch_size, 1]

        Returns
            embedded_q: embedded questions, shape = [batch_size, q_hidden(32)]
        """
        with tf.variable_scope(scope):
            embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            q_input = tf.unstack(embedded_q_word, num=self.q_max_len, axis=1)
            lstm_cell = rnn.BasicLSTMCell(self.word_embed, reuse=reuse)
            outputs, _ = rnn.static_rnn(lstm_cell, q_input, dtype=tf.float32)

            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            index = tf.range(0, self.batch_size) * self.q_max_len + (q_real_len - 1)
            embedded_q = tf.gather(tf.reshape(outputs, [-1, self.word_embed]), index)
        return embedded_q

    def contextGRU(self, c, s_real_lens, reuse=False, scope="contextGRU"):

        def sentenceGRU(s,
                        s_real_len,
                        reuse=reuse,
                        scope="sentenceGRU"):
            """
            embedding sentence

            Arguments
                s: sentence (word index list), shape = [batch_size*130, 12]
                s_real_len: length of the sentence before zero padding, int32

            Returns
                embedded_s: embedded sentence, shape = [batch_size*130, 32]
            """
            with tf.variable_scope(scope):
                embedded_sentence_word = tf.nn.embedding_lookup(self.word_embed_matrix, s)
                s_input = tf.unstack(embedded_sentence_word, num=self.s_max_len, axis=1)
                gru_cell = rnn.GRUCell(self.word_embed, reuse=reuse)
                outputs, _ = rnn.static_rnn(gru_cell, s_input, dtype=tf.float32)
                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                index = tf.range(0, self.batch_size * self.c_max_len) * self.s_max_len + (s_real_len - 1)
                outputs = tf.gather(tf.reshape(outputs, [-1, self.word_embed]), index)
            return outputs

        """
        contextGRU without label
        
        Args:
            c: list of sentences, shape = [batch_size, 130, 12]
            s_real_lens: list of real length, shape = [batch_size, 130]

        Returns:
            c_embedded: list of embedded sentence, shape = [batch_size, 32] 130개
            len(c_embedded) = 130
        """
        with tf.variable_scope(scope):
            sentences = tf.reshape(c, shape=[-1, self.s_max_len])
            real_lens = tf.reshape(s_real_lens, shape=[-1])

            s_embedded = sentenceGRU(sentences, real_lens, reuse=reuse)
            c_embedded = tf.reshape(s_embedded, shape=[self.batch_size, self.c_max_len, self.word_embed])
            c_embedded = tf.unstack(c_embedded, axis=1)
        return c_embedded

    def questionGRU(self, q, q_real_len, reuse=False, scope="questionGRU"):
        """
        Args
            q: zero padded questions, shape=[batch_size, q_max_len]
            q_real_len: original question length, shape = [batch_size, 1]

        Returns
            embedded_q: embedded questions, shape = [batch_size, q_hidden(32)]
        """
        with tf.variable_scope(scope):
            embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            q_input = tf.unstack(embedded_q_word, num=self.q_max_len, axis=1)
            gru_cell = rnn.GRUCell(self.word_embed, reuse=reuse)
            outputs, _ = rnn.static_rnn(gru_cell, q_input, dtype=tf.float32)

            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            index = tf.range(0, self.batch_size) * (self.q_max_len) + (q_real_len - 1)
            embedded_q = tf.gather(tf.reshape(outputs, [-1, self.word_embed]), index)
        return embedded_q

    def add_label(self, embedded_c, l, scope='add_label'):
        """
        Args:
            embedded_c: list of embedded sentence, shape = [batch_size, 32] 130개
            l: labels, shape = [batch_size, 130]
            embedded_q: embedded questions, shape = [batch_size, q_hidden(32)]

        Returns:
            concat_result: shape = [batch_size, 33] 130개
            embedded_q: add 0, shape = [batch_size, 33]
        """
        with tf.variable_scope(scope):
            embedded_c = tf.stack(embedded_c, axis=1)
            labels = tf.expand_dims(l, axis=2)  # [batch_size, 130, 1]
            concated = tf.concat([embedded_c, labels], axis=2)
            # embedded_q = tf.concat([embedded_q, tf.zeros(shape=[self.batch_size, 1])], axis=1)
        return tf.unstack(concated, axis=1)  # , embedded_q

    def add_label_last(self, embedded_c, l, embedded_q, scope='add_label_last'):
        """
        Args:
            embedded_c: list of embedded sentence, shape = [batch_size, 32] 130개
            l: labels, shape = [batch_size, 130]
            embedded_q: embedded questions, shape = [batch_size, q_hidden(32)]

        Returns:
            concat_result: shape = [batch_size, 33] 130개
            embedded_q: add 0, shape = [batch_size, 33]
        """
        with tf.variable_scope(scope):
            embedded_c = tf.stack(embedded_c, axis=1)
            labels = tf.expand_dims(l, axis=2)  # [batch_size, 130, 1]
            concated = tf.concat([embedded_c, labels], axis=2)
            embedded_q = tf.concat([embedded_q, tf.zeros(shape=[self.batch_size, 1])], axis=1)
        return tf.unstack(concated, axis=1), embedded_q

    def a_nn(self, z, phase=True, reuse=True, scope="", with_beta=True):
        """
        Args:
            z ([batch_size, c_max_len, 2*word_embed])
        """
        z = tf.reshape(z, shape=[self.batch_size * self.c_max_len, -1])
        units = [256, 128, 1]
        with tf.variable_scope(scope, reuse=reuse):
            a_1 = self.batch_norm_wrapper(z, units[0], phase=phase, scope='a_nn_1')
            a_2 = self.batch_norm_wrapper(a_1, units[1], phase=phase, scope='a_nn_2')
            a_3 = self.batch_norm_wrapper(a_2, units[2], phase=phase, scope='a_nn_3')
            Z = tf.reshape(a_3, shape=[self.batch_size, units[2], self.c_max_len])
            if with_beta:
                init = tf.contrib.layers.xavier_initializer()
                strength = tf.tile(1 + tf.nn.softplus(tf.get_variable('strength', shape=[1, 1, 1], initializer=init)),
                                   tf.constant([self.batch_size, 1, self.c_max_len]))
                alpha = tf.nn.softmax(strength * Z, name='alpha')  # [batch_size, 1, c_max_len]
                # alpha = tf.nn.relu(strength * Z, name='alpha')
            else:
                alpha = tf.nn.softmax(Z, name='alpha_without_beta')  # [batch_size, 1, c_max_len]
                # alpha = tf.nn.relu(Z, name='alpha_without_beta')
        return alpha

    def attention_with_original_no_cosine_no_q(self, prev_alpha, ss, embedded_c, phase=True, reuse=True, scope="",
                                               with_beta=True):
        """
        calculate attention without q, no cosine similarity, neural network
        Args:
            ss (decoder output, [batch_size, word_embed])
            embedded_c ([batch_size, c_max_len, word_embed])
        """
        ss = tf.tile(tf.expand_dims(ss, axis=1), tf.constant([1, self.c_max_len, 1]))
        embedded_c = (1 - tf.transpose(prev_alpha, [0, 2, 1])) * embedded_c
        z = tf.concat([ss, embedded_c], axis=2)
        alpha = self.a_nn(z, phase=phase, reuse=reuse, scope=scope, with_beta=with_beta)
        c = tf.squeeze(tf.matmul(alpha, embedded_c))  # [batch_size, word_embed]
        return alpha, c

    def get_mix_with_nn(self, embedded_c, embedded_q, phase=True, with_beta=True):
        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention_with_original_no_cosine_no_q(alpha_0, embedded_q, embedded_c, phase=phase,
                                                                   reuse=False, scope="a_1", with_beta=with_beta)
        alpha_2, m_2 = self.attention_with_original_no_cosine_no_q(alpha_1, m_1, embedded_c, phase=phase,
                                                                   reuse=False, scope="a_2", with_beta=with_beta)
        ss = [m_1, m_2]
        alphas = [alpha_1, alpha_2, alpha_1, alpha_2, alpha_1, alpha_2, alpha_1, alpha_2, alpha_1, alpha_2]
        return ss, alphas

    def concat_with_q_last(self, last_ss, embedded_q):
        return tf.concat([last_ss, embedded_q], axis=1)  # [batch_size, word_embed*2]

    def batch_norm_relu(self, inputs, output_shape, phase=True, scope=None, activation=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            h1 = fully_connected(inputs, output_shape, activation_fn=None, scope='dense')
            h2 = batch_norm(h1, decay=0.95, center=True, scale=True, is_training=phase, scope='bn',
                            updates_collections=None)
            if activation:
                o = tf.nn.relu(h2, 'relu')
            else:
                o = h2
        return o

    def batch_norm_wrapper(self, inputs, output_shape, phase, scope, activation=True):
        return tf.cond(phase, lambda: self.batch_norm_relu(inputs, output_shape, phase=True, scope=scope,
                                                           activation=activation, reuse=False),
                       lambda: self.batch_norm_relu(inputs, output_shape, phase=False, scope=scope,
                                                    activation=activation, reuse=True))

    def g_theta_last(self, RN_input, scope='g_theta', reuse=True, phase=True):
        """
        Args:
            RN_input: [o_i, o_j, q], shape = [batch_size*45, 240]
        Returns:
            g_output: shape = [45, batch_size, 256]
        """
        g_units = [256, 256]
        with tf.variable_scope(scope, reuse=reuse) as scope:
            g_1 = self.batch_norm_wrapper(RN_input, g_units[0], scope='g_1', phase=phase)
            g_2 = self.batch_norm_wrapper(g_1, g_units[1], scope='g_2', phase=phase)
        g_output = tf.reshape(g_2, shape=[-1, self.batch_size, g_units[-1]])
        return g_output

    def f_phi_last(self, g, scope="f_phi", reuse=True, phase=True):
        """
        Args:
            g: g_theta result, shape = [45, batch_size, 256]
        Returns:
            f_output: shape = [batch_size, 159]
        """
        # f_input = tf.reduce_sum(g, axis=0)
        f_units = [256, 512, self.answer_vocab_size]
        with tf.variable_scope(scope, reuse=reuse) as scope:
            f_1 = self.batch_norm_wrapper(g, f_units[0], scope="f_1", phase=phase)
            f_2 = self.batch_norm_wrapper(f_1, f_units[1], scope="f_2", phase=phase)
            f_3_1 = self.batch_norm_wrapper(f_2, f_units[2], activation=False, scope="f_3_1", phase=phase)
            f_3_2 = self.batch_norm_wrapper(f_2, f_units[2], activation=False, scope="f_3_2", phase=phase)
            f_3_3 = self.batch_norm_wrapper(f_2, f_units[2], activation=False, scope="f_3_3", phase=phase)
            f_3 = tf.concat([f_3_1, f_3_2, f_3_3], axis=1)
        return f_3

    def get_corr_acc_loss_last(self, pred, answer, answer_num):
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

    def g_theta(self, z, phase=True, activation=tf.nn.relu, reuse=True, with_softmax=True, with_beta=True, scope=""):
        """
        Args:
            z ([batch_size, c_max_len, 2*word_embed])
        """
        z = tf.reshape(z, shape=[self.batch_size * self.c_max_len, -1])
        g_units = self.g_theta_layers
        assert g_units[-1] == 1  # attention should be ended with layer sized 1
        with tf.variable_scope(scope, reuse=reuse):
            a = self.bn_module(z, g_units, phase=phase, activation=activation)
            a = tf.reshape(a, shape=[self.batch_size, g_units[-1], self.c_max_len])
            if with_softmax:
                if with_beta:
                    init = tf.contrib.layers.xavier_initializer(seed=self.seed)
                    strength = tf.tile(
                        1 + tf.nn.softplus(tf.get_variable('strength', shape=[1, 1, 1], initializer=init)),
                        tf.constant([self.batch_size, 1, self.c_max_len]))
                    alpha = tf.nn.softmax(strength * a, name='alpha')  # [batch_size, 1, c_max_len]
                else:
                    alpha = tf.nn.softmax(a, name='alpha_without_beta')  # [batch_size, 1, c_max_len]
            else:
                alpha = a
        return alpha

    def attention(self, prev_alpha, ss, embedded_c, phase=True, activation=tf.nn.relu, reuse=True, scope="",
                  with_softmax=True, with_beta=True):
        """
        calculate attention with g_theta

        Args:
            ss (decoder output, [batch_size, word_embed])
            embedded_c ([batch_size, c_max_len, word_embed])
        """
        ss = tf.tile(tf.expand_dims(ss, axis=1), tf.constant([1, self.c_max_len, 1]))
        embedded_c = (1 - tf.transpose(prev_alpha, [0, 2, 1])) * embedded_c
        z = tf.concat([ss, embedded_c], axis=2)
        alpha = self.g_theta(z, phase=phase, activation=activation, reuse=reuse, scope=scope, with_softmax=with_softmax,
                             with_beta=with_beta)
        c = tf.squeeze(tf.matmul(alpha, embedded_c))  # [batch_size, word_embed]
        return alpha, c

    def hop_2(self, embedded_c, embedded_q, phase=True, activation=tf.nn.relu, with_softmax=True, with_beta=True,
              with_alpha=True):
        """
        two hops
        1 hop: [context, embedded_q] --> alpha_1, m_1
        2 hop: [context, m_1] --> alpha_2, m_2 # TODO: this can be tested with different conditions
        """

        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, activation=activation,
                                      reuse=False, scope="a_1", with_softmax=with_softmax, with_beta=with_beta)
        if with_alpha:
            alpha_2, m_2 = self.attention(alpha_1, m_1, embedded_c, phase=phase, activation=activation,
                                          reuse=False, scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        else:
            alpha_2, m_2 = self.attention(alpha_0, m_1, embedded_c, phase=phase, activation=activation,
                                          reuse=False, scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        ss = [m_1, m_2]
        alphas = [alpha_1, alpha_2]
        return ss, alphas

    def hop_3(self, embedded_c, embedded_q, phase=True, activation=tf.nn.relu, with_softmax=True,
              with_beta=True):
        """
        three hops
        1 hop: [context, embedded_q] --> alpha_1, m_1
        2 hop: [(1-alpha_1)context, m_1] --> alpha_2, m_2 # TODO: this can be tested with different conditions
        3 hop: [(1-alpha_1)(1-alpha_2)context, m_2] --> alpha_3, m_3
        """
        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, activation=activation,
                                      reuse=False, scope="a_1", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_1, [0, 2, 1])) * embedded_c
        alpha_2, m_2 = self.attention(alpha_0, m_1, embedded_c, phase=phase, activation=activation, reuse=False,
                                      scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_2, [0, 2, 1])) * embedded_c
        alpha_3, m_3 = self.attention(alpha_0, m_2, embedded_c, phase=phase, activation=activation, reuse=False,
                                      scope="a_3", with_softmax=with_softmax, with_beta=with_beta)
        ss = [m_1, m_2, m_3]
        alphas = [alpha_1, alpha_2, alpha_3]
        return ss, alphas

    def concat_with_q(self, last_ss, embedded_q):
        return tf.concat([last_ss, embedded_q], axis=1)  # [batch_size, word_embed*2]

    def f_phi_fc(self, g, scope="f_phi", reuse=True, phase=True):
        """
        Args:
            mixed: [last m, embedded_q] 

        Returns:
            f_output: shape = [batch_size, 159]
        """
        f_units = self.f_phi_layers
        with tf.variable_scope(scope, reuse=reuse) as scope:
            f_output = self.bn_module(g, f_units, phase=phase, activation=tf.nn.relu)
            with tf.variable_scope("pred_1", reuse=reuse):
                pred_1 = self.fc_module(f_output, [self.answer_vocab_size], activation=None, phase=phase)
            with tf.variable_scope("pred_2", reuse=reuse):
                pred_2 = self.fc_module(f_output, [self.answer_vocab_size], activation=None, phase=phase)
            with tf.variable_scope("pred_3", reuse=reuse):
                pred_3 = self.fc_module(f_output, [self.answer_vocab_size], activation=None, phase=phase)
        pred = tf.concat([pred_1, pred_2, pred_3], axis=1)
        return pred

    def f_phi(self, g, scope="f_phi", reuse=True, phase=True):
        """
        Args:
            mixed: [last m, embedded_q] 

        Returns:
            f_output: shape = [batch_size, 159]
        """
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

    def convert_to_RN_input(self, embedded_c, embedded_q):
        """
        Args
            embedded_c: output of contextLSTM, 20 length list of embedded sentences
            embedded_q: output of questionLSTM, embedded question
        Returns
            RN_input: input for RN g_theta, shape = [batch_size*190, (52+52+32)]
            considered batch_size and all combinations
        """
        # 20 combination 2 --> total 190 object pairs
        object_pairs = list(itertools.combinations(embedded_c, 2))
        # concatenate with question
        RN_inputs = []
        for object_pair in object_pairs:
            RN_input = tf.concat([object_pair[0], object_pair[1], embedded_q], axis=1)
            RN_inputs.append(RN_input)

        return tf.concat(RN_inputs, axis=0)

    def g_theta_RN(self, RN_input, phase=True, activation=tf.nn.relu, reuse=True, scope=""):
        """
        Args:
            z ([batch_size, c_max_len, 2*word_embed])
        """
        g_units = self.g_theta_layers
        with tf.variable_scope(scope, reuse=reuse):
            g_output = self.bn_module(RN_input, g_units, phase=phase, activation=activation)
            g_output = tf.reshape(g_output, shape=[-1, self.batch_size, g_units[-1]])
        return g_output

    def f_phi_RN(self, g, scope="f_phi", reuse=True, phase=True):
        """
        Args:
            mixed: [last m, embedded_q] 

        Returns:
            f_output: shape = [batch_size, 159]
        """
        f_input = tf.reduce_sum(g, axis=0)
        f_units = self.f_phi_layers
        with tf.variable_scope(scope, reuse=reuse) as scope:
            f_output = self.bn_module(f_input, f_units, phase=phase, activation=tf.nn.relu)
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

    def get_corr_acc_loss_cl(self, pred, answer, answer_num):
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
        return correct, accuracy, loss, loss_before
