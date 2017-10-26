import tensorflow as tf

from module import Module


class Model():
    def __init__(self,
                 config,
                 seed=9,
                 word_embed=32,
                 answer_vocab_size=159):
        self.config = config
        self.batch_size = config['batch_size']
        self.c_max_len = config['c_max_len']  # 130
        self.s_max_len = config['s_max_len']  # 12
        self.q_max_len = config['q_max_len']  # 12
        self.seed = seed
        self.word_embed = word_embed
        self.answer_vocab_size = config['vocab_size']
        self.context = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len, self.s_max_len],
            name="context"
        )
        self.sentence_real_len = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len],
            name="sentence_real_length"
        )
        self.question = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.q_max_len],
            name="question"
        )
        self.question_real_len = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size],
            name="question_real_length"
        )
        self.label = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, self.c_max_len],
            name='label'
        )
        self.answer = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, 3, self.answer_vocab_size],
            name="answer"
        )
        self.answer_num = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, 3],
            name="answer_num"
        )
        self.hint = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len],
            name='hint'
        )
        self.is_training = tf.placeholder(
            dtype=tf.bool,
            name="is_training"
        )

        self.g_theta_layers = None
        self.f_phi_layers = None
        self.alpha_1 = None
        self.alpha_2 = None
        self.alpha_3 = None

    def vRN(self):
        self.g_theta_layers = [256,256,256,256]
        self.f_phi_layers = [256,512]
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        RN_input = md.convert_to_RN_input(embedded_c, embedded_q)
        g_output = md.g_theta_RN(RN_input, phase = self.is_training, scope = 'g_theta_rn', reuse = False)
        prediction = md.f_phi_RN(g_output, reuse = False, phase = self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        return pred, correct, accuracy, loss


    def vfinal(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]  # attention
        self.f_phi_layers = [512, 512]  # answer
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.relu, with_softmax=True,
                             with_beta=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss

    def vfinal_hop3(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        3 hop
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]  # attention
        self.f_phi_layers = [512, 512]  # answer
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        m, alphas = md.hop_3(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.relu, with_softmax=True,
                             with_beta=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        self.alpha_3 = alphas[2]
        return pred, correct, accuracy, loss

    def vlast(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop
        add_label with / 100
        use last ss
        f_phi
        """
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)        
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c, embedded_q = md.add_label_last(embedded_c, self.label, embedded_q)
        m, alphas = md.get_mix_with_nn(embedded_c, embedded_q, phase=self.is_training, with_beta=True)
        input_ = md.concat_with_q_last(m[-1], embedded_q)
        prediction = md.f_phi_last(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss_last(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss

    def vfinal_fc(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]  # attention
        self.f_phi_layers = [256, 512]  # answer
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.relu, with_softmax=True,
                             with_beta=True)  # TODO: 원본)) with_softmax=True, tf.nn.relu, with_beta=True
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi_fc(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss


    def vfinal_tanh(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]  # attention
        self.f_phi_layers = [256, 512]  # answer
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.tanh, with_beta=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss

    def vfinal_cl(self):
        """
        curriculum learning
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]
        self.f_phi_layers = [256, 512]
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = self.md.add_label(embedded_c, self.label)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.relu, with_beta=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss, loss_before = md.get_corr_acc_loss_cl(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss, loss_before

    def vno_alpha_relu(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop (not using alpha, 1-alpha)
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]  # attention
        self.f_phi_layers = [256, 512]  # answer
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.relu, with_beta=True,
                             with_alpha=False)
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss

    def vno_alpha_tanh(self):
        """
        change attention calculation with nn, no duplicate, no cosine sim
        2 hop (not using alpha, 1-alpha)
        add_label with / 100
        use last ss
        f_phi
        """
        self.g_theta_layers = [256, 128]  # attention
        self.f_phi_layers = [256, 512]  # answer
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers, seed=self.seed, word_embed=self.word_embed,
                    answer_vocab_size=self.answer_vocab_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.tanh, with_beta=True,
                             with_alpha=False)
        input_ = md.concat_with_q(m[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss

