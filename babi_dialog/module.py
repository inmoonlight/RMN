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
        # init = xavier_initializer(seed=self.seed)
        # word_embed_matrix = tf.get_variable(name='word_embed_matrix', shape=[self.vocab_size, self.word_embed_dim], initializer=init)
        word_embed_matrix = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.word_embed_dim], minval=-1,
                                                          maxval=1, seed=self.seed), name='word_embed_matrix')
        return word_embed_matrix

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

    def contextLSTM(self, c, s_real_lens, reuse=False, with_embed_matrix=True, scope="contextLSTM"):
        def sentenceLSTM(s,
                         s_real_len,
                         reuse=reuse,
                         with_embed_matrix=with_embed_matrix,
                         scope="sentenceLSTM"):
            """
            embedding sentence

            Args:
                s: sentence (word index list), shape = [batch_size*c_max_len, 12]
                s_real_len: length of the sentence before zero padding, int32

            Returns:
                embedded_s: embedded sentence, shape = [batch_size*c_max_len, 32]
            """
            with tf.variable_scope(scope):
                if with_embed_matrix:
                    embedded_sentence_word = tf.nn.embedding_lookup(self.word_embed_matrix, s)
                else:
                    embedded_sentence_word = tf.one_hot(indices=s, depth=self.vocab_size)
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
            c_embedded: list of embedded sentence, shape = [batch_size, 32] 130개
            len(c_embedded) = 130
        """
        with tf.variable_scope(scope):
            sentences = tf.reshape(c, shape=[-1, self.s_max_len])
            real_lens = tf.reshape(s_real_lens, shape=[-1])

            s_embedded = sentenceLSTM(sentences, real_lens, reuse=reuse)
            c_embedded = tf.reshape(s_embedded, shape=[self.batch_size, self.c_max_len, self.hidden_dim])
            c_embedded = tf.unstack(c_embedded, axis=1)
        return c_embedded

    def contextSum(self, c, scope='contextSum', with_embed_matrix=True, with_position_encoding=False):
        sentences = tf.reshape(c, shape=[-1, self.s_max_len])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_s_word = tf.nn.embedding_lookup(self.word_embed_matrix,
                                                         sentences)  # [batch_size*context_size, s_max_len, embedding_dim]
            else:
                embedded_s_word = tf.one_hot(indices=sentences, depth=self.vocab_size)

            if with_position_encoding:
                s_sum = tf.reduce_sum(embedded_s_word * self.encoding,
                                      axis=1)  # [batch_size*context_size, embedding_dim]
            else:
                s_sum = tf.reduce_sum(embedded_s_word, axis=1)

            c_embedded = tf.reshape(s_sum, shape=[self.batch_size, self.c_max_len, -1])
        return tf.unstack(c_embedded, axis=1)

    def contextConcat(self, c, scope='contextConcat', with_embed_matrix=True, with_position_encoding=False):
        sentences = tf.reshape(c, shape=[-1, self.s_max_len])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_s_word = tf.nn.embedding_lookup(self.word_embed_matrix,
                                                         sentences)  # [batch_size*context_size, s_max_len, embedding_dim]
            else:
                embedded_s_word = tf.one_hot(indices=sentences, depth=self.vocab_size)

            if with_position_encoding:
                s_concat = tf.concat(tf.unstack(embedded_s_word * self.encoding, axis=1),
                                     axis=1)  # [batch_size*context_size, s_max_len*embedding_dim]
            else:
                s_concat = tf.concat(tf.unstack(embedded_s_word, axis=1), axis=1)

            c_embedded = tf.reshape(s_concat, shape=[self.batch_size, self.c_max_len, -1])
        return tf.unstack(c_embedded, axis=1)

    def questionLSTM(self, q, q_real_len, reuse=False, with_embed_matrix=True, scope="questionLSTM"):
        """
        Args
            q: zero padded questions, shape=[batch_size, q_max_len]
            q_real_len: original question length, shape = [batch_size, 1]

        Returns
            embedded_q: embedded questions, shape = [batch_size, q_hidden(32)]
        """
        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            else:
                embedded_q_word = tf.one_hot(indices=q, depth=self.vocab_size)
            q_input = tf.unstack(embedded_q_word, num=self.q_max_len, axis=1)
            lstm_cell = rnn.BasicLSTMCell(self.hidden_dim, reuse=reuse)
            outputs, _ = rnn.static_rnn(lstm_cell, q_input, dtype=tf.float32)

            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            index = tf.range(0, self.batch_size) * self.q_max_len + (q_real_len - 1)
            q_embedded = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
        return q_embedded

    def questionSum(self, q, scope='questionSum', with_embed_matrix=True, with_position_encoding=False):
        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_q_word = tf.nn.embedding_lookup(self.word_embed_matrix, q)
            else:
                embedded_q_word = tf.one_hot(indices=q, depth=self.vocab_size)

            if with_position_encoding:
                q_sum = tf.reduce_sum(embedded_q_word * self.encoding, axis=1)  # [batch_size, embedding_dim]
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
                q_concat = tf.concat(tf.unstack(embedded_q_word * self.encoding, axis=1),
                                     axis=1)  # [batch_size, embedding_dim]
            else:
                q_concat = tf.concat(tf.unstack(embedded_q_word, axis=1), axis=1)
        return q_concat

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
                gru_cell = rnn.GRUCell(self.hidden_dim, reuse=reuse)
                outputs, _ = rnn.static_rnn(gru_cell, s_input, dtype=tf.float32)
                outputs = tf.stack(outputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                index = tf.range(0, self.batch_size * self.c_max_len) * self.s_max_len + (s_real_len - 1)
                outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
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
            c_embedded = tf.reshape(s_embedded, shape=[self.batch_size, self.c_max_len, self.hidden_dim])
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
            gru_cell = rnn.GRUCell(self.hidden_dim, reuse=reuse)
            outputs, _ = rnn.static_rnn(gru_cell, q_input, dtype=tf.float32)

            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            index = tf.range(0, self.batch_size) * self.q_max_len + (q_real_len - 1)
            embedded_q = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
        return embedded_q

    def bn_module(self, input_, layers, phase, activation=tf.nn.relu):
        outputs = [input_]
        for layer_dim in layers:
            fc_output = fully_connected(outputs[-1], layer_dim, activation_fn=None)
            bn_output = batch_norm(fc_output, decay=0.95, center=True, scale=True, is_training=phase,
                                   updates_collections=None, activation_fn=activation)
            outputs.append(bn_output)
        return outputs[-1]

    def ln_module(self, input_, layers, phase=True, activation=tf.nn.relu):
        outputs = [input_]
        for layer_dim in layers:
            fc_output = fully_connected(outputs[-1], layer_dim, activation_fn=None)
            ln_output = tf.cond(phase,
                                lambda: layer_norm(fc_output, center=True, scale=True, trainable=True,
                                                   activation_fn=activation),
                                lambda: layer_norm(fc_output, center=True, scale=True, trainable=False,
                                                   activation_fn=activation))
            outputs.append(ln_output)
        return outputs[-1]

    def fc_module(self, input_, layers, phase, activation=tf.nn.relu):
        outputs = [input_]
        for layer_dim in layers:
            fc_output = fully_connected(outputs[-1], layer_dim, activation_fn=activation)
            outputs.append(fc_output)
        return outputs[-1]

    def g_theta(self, z, norm='bn', phase=True, activation=tf.nn.relu, reuse=True, with_softmax=True, with_beta=True,
                scope=""):
        """
        Args:
            z ([batch_size, c_max_len, 2*word_embed])
        """
        z = tf.reshape(z, shape=[self.batch_size * self.c_max_len, -1])
        g_units = self.g_theta_layers
        assert g_units[-1] == 1  # attention should be ended with layer sized 1
        with tf.variable_scope(scope, reuse=reuse):
            if norm == 'bn':
                a = self.bn_module(z, g_units, phase=phase, activation=activation)
            elif norm == 'fc':
                a = self.fc_module(z, g_units, phase=phase, activation=activation)
            elif norm == 'ln':
                a = self.ln_module(z, g_units, phase=phase, activation=activation)
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

    def attention(self, prev_alpha, ss, embedded_c, phase=True, norm='bn', activation=tf.nn.relu, reuse=True,
                  scope="", with_softmax=True, with_beta=True):
        """
        calculate attention with g_theta

        Args:
            ss (decoder output, [batch_size, word_embed])
            embedded_c ([batch_size, c_max_len, word_embed])
        """
        ss = tf.tile(tf.expand_dims(ss, axis=1), tf.constant([1, self.c_max_len, 1]))
        embedded_c = (1 - tf.transpose(prev_alpha, [0, 2, 1])) * embedded_c
        z = tf.concat([ss, embedded_c], axis=2)
        alpha = self.g_theta(z, norm=norm, phase=phase, activation=activation, reuse=reuse, scope=scope,
                             with_softmax=with_softmax, with_beta=with_beta)
        c = tf.squeeze(tf.matmul(alpha, embedded_c))  # [batch_size, word_embed]
        return alpha, c

    def hop_1(self, embedded_c, embedded_q, phase=True, norm='bn', activation=tf.nn.relu, with_softmax=True,
              with_beta=True, with_alpha=True):
        """
        one hop
        1 hop: [context, embedded_q] --> alpha_1, m_1
        """

        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, norm=norm,
                                      activation=activation,
                                      reuse=False, scope="a_1", with_softmax=with_softmax, with_beta=with_beta)
        ss = [m_1]
        alphas = [alpha_1]
        return ss, alphas

    def hop_2(self, embedded_c, embedded_q, phase=True, norm='bn', activation=tf.nn.relu, with_softmax=True,
              with_beta=True,
              with_alpha=True):
        """
        two hops
        1 hop: [context, embedded_q] --> alpha_1, m_1
        2 hop: [context, m_1] --> alpha_2, m_2 # TODO: this can be tested with different conditions
        """

        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, norm=norm,
                                      activation=activation,
                                      reuse=False, scope="a_1", with_softmax=with_softmax, with_beta=with_beta)
        if with_alpha:
            alpha_2, m_2 = self.attention(alpha_1, m_1, embedded_c, phase=phase, norm=norm, activation=activation,
                                          reuse=False, scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        else:
            alpha_2, m_2 = self.attention(alpha_0, m_1, embedded_c, phase=phase, norm=norm, activation=activation,
                                          reuse=False, scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        ss = [m_1, m_2]
        alphas = [alpha_1, alpha_2]
        return ss, alphas

    def hop_3(self, embedded_c, embedded_q, phase=True, norm='bn', activation=tf.nn.relu, with_softmax=True,
              with_beta=True, with_alpha=True):
        """
        three hops
        1 hop: [context, embedded_q] --> alpha_1, m_1
        2 hop: [(1-alpha_1)context, m_1] --> alpha_2, m_2 # TODO: this can be tested with different conditions
        3 hop: [(1-alpha_1)(1-alpha_2)context, m_2] --> alpha_3, m_3
        """
        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, norm=norm,
                                      activation=activation,
                                      reuse=False, scope="a_1", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_1, [0, 2, 1])) * embedded_c
        alpha_2, m_2 = self.attention(alpha_0, m_1, embedded_c, phase=phase, norm=norm, activation=activation,
                                      reuse=False, scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_2, [0, 2, 1])) * embedded_c
        alpha_3, m_3 = self.attention(alpha_0, m_2, embedded_c, phase=phase, norm=norm, activation=activation,
                                      reuse=False, scope="a_3", with_softmax=with_softmax, with_beta=with_beta)
        ss = [m_1, m_2, m_3]
        alphas = [alpha_1, alpha_2, alpha_3]
        return ss, alphas

    def hop_4(self, embedded_c, embedded_q, phase=True, norm='bn', activation=tf.nn.relu, with_softmax=True,
              with_beta=True, with_alpha=True):
        """
        four hops
        1 hop: [context, embedded_q] --> alpha_1, m_1
        2 hop: [(1-alpha_1)context, m_1] --> alpha_2, m_2 # TODO: this can be tested with different conditions
        3 hop: [(1-alpha_1)(1-alpha_2)context, m_2] --> alpha_3, m_3
        4 hop: [(1-alpha_1)(1-alpha_2)(1_alpha_3)context, m_2] --> alpha_4, m_4
        """
        alpha_0 = tf.zeros([self.batch_size, 1, self.c_max_len])
        embedded_c = tf.stack(embedded_c, axis=1)  # [batch_size, c_max_len, word_embed]
        alpha_1, m_1 = self.attention(alpha_0, embedded_q, embedded_c, phase=phase, norm=norm,
                                      activation=activation,
                                      reuse=False, scope="a_1", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_1, [0, 2, 1])) * embedded_c
        alpha_2, m_2 = self.attention(alpha_0, m_1, embedded_c, phase=phase, norm=norm, activation=activation,
                                      reuse=False, scope="a_2", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_2, [0, 2, 1])) * embedded_c
        alpha_3, m_3 = self.attention(alpha_0, m_2, embedded_c, phase=phase, norm=norm, activation=activation,
                                      reuse=False, scope="a_3", with_softmax=with_softmax, with_beta=with_beta)
        embedded_c = (1 - tf.transpose(alpha_3, [0, 2, 1])) * embedded_c
        alpha_4, m_4 = self.attention(alpha_0, m_3, embedded_c, phase=phase, norm=norm, activation=activation,
                                      reuse=False, scope="a_4", with_softmax=with_softmax, with_beta=with_beta)
        ss = [m_1, m_2, m_3, m_4]
        alphas = [alpha_1, alpha_2, alpha_3, alpha_4]
        return ss, alphas

    def concat_with_q(self, last_ss, embedded_q):
        return tf.concat([last_ss, embedded_q], axis=1)  # [batch_size, word_embed*2]

    def convert_to_RN_input(self, embedded_c, embedded_q):
        """
        Args
            embedded_c: output of contextLSTM, 20 length list of embedded sentences
            embedded_q: output of questionLSTM, embedded question

        Returns
            RN_input: input for RN g_theta, shape = [batch_size*190, (52+52+32)]
            considered batch_size and all combinations
        """
        # combination 2 --> total 190 object pairs
        object_pairs = list(itertools.combinations(embedded_c, 2))
        # concatenate with question
        RN_inputs = []
        for object_pair in object_pairs:
            RN_input = tf.concat([object_pair[0], object_pair[1], embedded_q], axis=1)
            RN_inputs.append(RN_input)
        return tf.concat(RN_inputs, axis=0)

    def g_theta_RN(self, RN_input, scope='g_theta_RN', reuse=True, phase=True):
        g_units = self.g_theta_layers
        with tf.variable_scope(scope, reuse=reuse):
            g_output = self.bn_module(RN_input, g_units, phase=phase, activation=tf.nn.relu)
        g_output = tf.reshape(g_output, shape=[-1, self.batch_size, g_units[-1]])
        return g_output

    def f_phi(self, g, norm='bn', activation=tf.nn.relu, scope="f_phi", reuse=True, with_embed_matrix=True,
              is_concat=True, use_match=False, phase=True):
        """
        Args:
            mixed: [last m, embedded_q]

        Returns:
            f_output: shape = [batch_size, 159]
        """
        f_units = self.f_phi_layers
        with tf.variable_scope(scope, reuse=reuse) as scope:
            if norm == 'bn':
                f_output = self.bn_module(g, f_units, phase=phase, activation=activation)
            elif norm == 'fc':
                f_output = self.fc_module(g, f_units, phase=phase, activation=activation)
            elif norm == 'ln':
                f_output = self.ln_module(g, f_units, phase=phase, activation=activation)
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

    def f_phi_RN(self, g, norm='bn', scope='f_phi_RN', reuse=True, phase=True):
        f_units = self.f_phi_layers
        f_input = tf.reduce_sum(g, axis=0)
        with tf.variable_scope(scope, reuse=reuse):
            if norm == 'bn':
                f_output = self.bn_module(f_input, f_units, phase=phase, activation=tf.nn.relu)
            elif norm == 'ln':
                f_output = self.ln_module(f_input, f_units, phase=phase, activation=tf.nn.relu)
            with tf.variable_scope("pred", reuse=reuse):
                pred = self.fc_module(f_output, [self.word_embed_dim], activation=None, phase=phase)
        return pred

    def answerLSTM(self, a, a_real_lens, reuse=False, with_embed_matrix=True, scope='answerLSTM'):
        """
        Args
            a: zero padded answers, shape=[batch_size, answer_size, 5]
            a_real_lens: original answer length, shape = [batch_size, answer_size, 1]

        Returns
            embedded_a: embedded answer, shape = [batch_size, answer_size, hidden(32)]
        """
        answers = tf.reshape(a, shape=[-1, self.a_max_len])
        real_lens = tf.reshape(a_real_lens, shape=[-1])

        with tf.variable_scope(scope):
            if with_embed_matrix:
                embedded_a_word = tf.nn.embedding_lookup(self.word_embed_matrix, answers)
            else:
                embedded_a_word = tf.one_hot(indices=answers, depth=self.vocab_size)
            a_input = tf.unstack(embedded_a_word, num=self.a_max_len, axis=1)
            lstm_cell = rnn.BasicLSTMCell(self.hidden_dim, reuse=reuse)
            outputs, _ = rnn.static_rnn(lstm_cell, a_input, dtype=tf.float32)

            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            index = tf.range(0, self.batch_size * self.cand_size) * self.a_max_len + (real_lens - 1)
            embedded_a = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)

        embedded_a = tf.reshape(embedded_a, shape=[self.batch_size, -1, self.hidden_dim])
        return embedded_a

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
        """
        answer_idx (batch_size): indicates answer idx among the candidates
        """
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
