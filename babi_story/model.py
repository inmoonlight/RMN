import tensorflow as tf

from module import Module


class RMN():
    def __init__(self,
                 config,
                 config_txt,
                 seed=9,
                 word_embed_dim=32,
                 hidden_dim=32):
        self.config = config
        self.config_txt = config_txt
        self.batch_size = config.batch_size
        self.c_max_len = config_txt['c_max_len']  # 130
        self.s_max_len = config_txt['s_max_len']  # 12
        self.q_max_len = config_txt['q_max_len']  # 12
        self.seed = seed
        self.word_embed_dim = word_embed_dim
        self.hidden_dim = hidden_dim
        self.answer_vocab_size = config_txt['vocab_size']
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
        self.is_training = tf.placeholder(
            dtype=tf.bool,
            name="is_training"
        )

    def run(self):
        g_theta_layers = [256, 128]  # attention component
        f_phi_layers = [512, 512]  # reasoning component
        g_theta_layers.append(1)  # attention should be ended with unit layer
        md = Module(self.config_txt, g_theta_layers, f_phi_layers, seed=self.seed, word_embed=self.word_embed_dim,
                    hidden_dim=self.hidden_dim, batch_size=self.batch_size)
        embedded_c = md.contextLSTM(self.context, self.sentence_real_len)
        embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.add_label(embedded_c, self.label)
        r, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.relu)
        input_ = md.concat_with_q(r[-1], embedded_q)
        prediction = md.f_phi(input_, reuse=False, phase=self.is_training)
        pred = tf.split(prediction, 3, axis=1)
        correct, accuracy, loss = md.get_corr_acc_loss(pred, self.answer, self.answer_num)
        return pred, correct, accuracy, loss
