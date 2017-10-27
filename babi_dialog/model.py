import tensorflow as tf

from module import Module
from input_ops import parse_config_txt
from util import log


class RMN:
    def __init__(self, config, seed=9):
        self.config = config
        self.batch_size = config.batch_size
        self.use_match = config.use_match
        self.embedding = config.embedding
        self.seed = seed
        config_txt = parse_config_txt('config_{}.txt'.format(config.task))
        self.c_max_len = config_txt['memory_size']
        self.s_max_len = config_txt['sentence_size']
        self.q_max_len = config_txt['sentence_size']
        self.a_max_len = config_txt['cand_sentence_size']
        self.cand_size = config_txt['cand_size']
        self._type_num = config_txt['_TYPE_NUM']
        self.context = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len, self.s_max_len],
            name='context'
        )
        self.context_real_len = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len],
            name="context_real_length"
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
        self.answer = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.cand_size, self.a_max_len],  # 모든 a
            name="answer"
        )
        self.answer_match = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, self.cand_size, self._type_num],
            name="answer_match"
        )
        self.answer_idx = tf.placeholder(
            dtype=tf.int64,
            shape=[self.batch_size],
            name="answer_idx"
        )
        self.is_training = tf.placeholder(
            dtype=tf.bool,
            name="is_training"
        )

        self.g_theta_layers = None
        self.f_phi_layers = None

    def run(self):
        if self.config.task == 1:
            self.g_theta_layers = [2048, 2048]
            self.f_phi_layers = [2048, 2048]
        elif self.config.task == 2:
            self.g_theta_layers = [1024, 1024]
            self.f_phi_layers = [1024, 1024]
        elif self.config.task == 3:
            self.g_theta_layers = [1024, 1024, 1024]
            self.f_phi_layers = [1024] * 3
        elif self.config.task == 4:
            self.g_theta_layers = [1024, 1024]
            self.f_phi_layers = [1024, 1024]
        elif self.config.task == 5:
            self.g_theta_layers = [4096, 4096]
            self.f_phi_layers = [4096, 4096]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        if self.embedding == 'sum':
            embedded_c = md.contextSum(self.context, with_embed_matrix=True)
            embedded_q = md.questionSum(self.question, with_embed_matrix=True)
        elif self.embedding == 'concat':
            embedded_c = md.contextConcat(self.context, with_embed_matrix=True)
            embedded_q = md.questionConcat(self.question, with_embed_matrix=True)
        m, alphas = md.hop_1(embedded_c, embedded_q, phase=self.is_training, activation=tf.nn.tanh)
        input_ = md.concat_with_q(m[-1], embedded_q)
        pred = md.f_phi(input_,activation=tf.nn.relu, reuse=False, with_embed_matrix=True,
                            is_concat=True, use_match=self.use_match, phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_match, self.answer_idx,
                                                                        with_position_encoding=False,
                                                                        with_embed_matrix=True, is_concat=True,
                                                                        use_match=self.use_match, is_cosine_sim=True)
        return pred, correct, accuracy, loss, sim_score, p, a