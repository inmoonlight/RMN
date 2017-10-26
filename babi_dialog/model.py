import tensorflow as tf

from module import Module
from input_ops import parse_config_txt
from util import log


class Model:
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
            shape=[self.batch_size],  # 특정 질문에 대한 a가 전체에서 몇번째 index인지
            name="answer_idx"
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
        self.alpha_4 = None

    def vRN(self):
        if self.config.task == 1:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 2:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 3:
            self.g_theta_layers = [1024, 1024]
            self.f_phi_layers = [1024, 1024]
        elif self.config.task == 4:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 5:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 6:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        # embedded_c = md.contextLSTM(self.context, self.sentence_real_len, with_embed_matrix=False)
        # embedded_q = md.questionLSTM(self.question, self.question_real_len)
        embedded_c = md.contextSum(self.context, with_embed_matrix=True)
        embedded_q = md.questionSum(self.question, with_embed_matrix=True)
        # embedded_c = md.add_label(embedded_c, self.speaker, self.order)
        RN_input = md.convert_to_RN_input(embedded_c, embedded_q)
        g_output = md.g_theta_RN(RN_input, phase=self.is_training, reuse=False)
        pred = md.f_phi_RN(g_output, norm='bn', reuse=False, phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_idx)
        return pred, correct, accuracy, loss, sim_score, p, a

    def vfinal_hop1(self):
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
        elif self.config.task == 6:
            self.g_theta_layers = [512, 512]
            self.f_phi_layers = [512, 512]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        # embedded_c = md.contextLSTM(self.context, self.context_real_len, with_embed_matrix=True)
        # embedded_q = md.questionLSTM(self.question, self.question_real_len, with_embed_matrix=True)
        if self.embedding == 'sum':
            embedded_c = md.contextSum(self.context, with_embed_matrix=True, with_position_encoding=True)
            embedded_q = md.questionSum(self.question, with_embed_matrix=True, with_position_encoding=True)
        elif self.embedding == 'concat':
            embedded_c = md.contextConcat(self.context, with_embed_matrix=True, with_position_encoding=False)
            embedded_q = md.questionConcat(self.question, with_embed_matrix=True, with_position_encoding=False)

        m, alphas = md.hop_1(embedded_c, embedded_q, phase=self.is_training, norm='bn', activation=tf.nn.tanh,
                             with_beta=True, with_alpha=True, with_softmax=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        pred = md.f_phi(input_, norm='bn', activation=tf.nn.relu, reuse=False, with_embed_matrix=True,
                            is_concat=True, use_match=self.use_match, phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_match, self.answer_idx,
                                                                        with_position_encoding=False,
                                                                        with_embed_matrix=True, is_concat=True,
                                                                        use_match=self.use_match, is_cosine_sim=True)
        self.alpha_1 = alphas[0]
        return pred, correct, accuracy, loss, sim_score, p, a

    def vfinal_hop2(self):
        if self.config.task == 1:
            self.g_theta_layers = [128] * 2
            self.f_phi_layers = [128] * 2
        elif self.config.task == 2:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 3:
            self.g_theta_layers = [1024]
            self.f_phi_layers = [1024, 1024]
        elif self.config.task == 4:
            self.g_theta_layers = [1024, 1024]
            self.f_phi_layers = [1024, 1024]
        elif self.config.task == 5:
            self.g_theta_layers = [4096, 4096]
            self.f_phi_layers = [2048, 2048]
        elif self.config.task == 6:
            self.g_theta_layers = [512, 512]
            self.f_phi_layers = [512, 512, 512]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        # embedded_c = md.contextLSTM(self.context, self.sentence_real_len, with_embed_matrix=True)
        # embedded_q = md.questionLSTM(self.question, self.question_real_len, with_embed_matrix=True)
        if self.embedding == 'sum':
            embedded_c = md.contextSum(self.context, with_embed_matrix=True, with_position_encoding=True)
            embedded_q = md.questionSum(self.question, with_embed_matrix=True, with_position_encoding=True)
        elif self.embedding == 'concat':
            embedded_c = md.contextConcat(self.context, with_embed_matrix=True, with_position_encoding=False)
            embedded_q = md.questionConcat(self.question, with_embed_matrix=True, with_position_encoding=False)

        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, norm='bn', activation=tf.nn.tanh,
                             with_softmax=True, with_beta=True, with_alpha=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        pred = md.f_phi(input_, norm='bn', activation=tf.nn.relu, reuse=False, with_embed_matrix=True,
                        is_concat=True, use_match=self.use_match, phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_match,
                                                                        self.answer_idx,
                                                                        with_position_encoding=False,
                                                                        with_embed_matrix=True, is_concat=True,
                                                                        use_match=self.use_match, is_cosine_sim=True)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss, sim_score, p, a

    def vfinal_hop3(self):
        if self.config.task == 1:
            self.g_theta_layers = [32]
            self.f_phi_layers = [32]
        elif self.config.task == 2:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 3:
            self.g_theta_layers = [1024]
            self.f_phi_layers = [1024, 1024]
        elif self.config.task == 4:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 5:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 6:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        # embedded_c = md.contextLSTM(self.context, self.sentence_real_len, with_embed_matrix=True)
        # embedded_q = md.questionLSTM(self.question, self.question_real_len, with_embed_matrix=True)
        # embedded_c = md.contextSum(self.context, with_embed_matrix=True)
        # embedded_q = md.questionSum(self.question, with_embed_matrix=True)
        embedded_c = md.contextConcat(self.context, with_embed_matrix=True, with_position_encoding=False)
        embedded_q = md.questionConcat(self.question, with_embed_matrix=True, with_position_encoding=False)
        # embedded_c = md.add_label(embedded_c, self.speaker, self.order)
        m, alphas = md.hop_3(embedded_c, embedded_q, phase=self.is_training, norm='bn', activation=tf.nn.tanh,
                             with_beta=True, with_alpha=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        pred = md.f_phi(input_, norm='bn', activation=tf.nn.relu, reuse=False, with_embed_matrix=True, is_concat=True,
                        phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_idx,
                                                                        with_position_encoding=True, is_concat=True)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        self.alpha_3 = alphas[2]
        return pred, correct, accuracy, loss, sim_score, p, a

    def vfinal_hop4(self):
        if self.config.task == 1:
            self.g_theta_layers = [32]
            self.f_phi_layers = [32]
        elif self.config.task == 2:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 3:
            self.g_theta_layers = [128] * 3
            self.f_phi_layers = [64, 64]
        elif self.config.task == 4:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 5:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 6:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        # embedded_c = md.contextLSTM(self.context, self.sentence_real_len, with_embed_matrix=True)
        # embedded_q = md.questionLSTM(self.question, self.question_real_len, with_embed_matrix=True)
        embedded_c = md.contextSum(self.context, with_embed_matrix=True)
        embedded_q = md.questionSum(self.question, with_embed_matrix=True)
        # embedded_c = md.add_label(embedded_c, self.speaker, self.order)
        m, alphas = md.hop_4(embedded_c, embedded_q, phase=self.is_training, norm='bn', activation=tf.nn.relu,
                             with_beta=True, with_alpha=True)
        input_ = md.concat_with_q(m[-1], embedded_q)
        pred = md.f_phi(input_, norm='bn', reuse=False, with_embed_matrix=True, phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_idx)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        self.alpha_3 = alphas[2]
        self.alpha_4 = alphas[3]
        return pred, correct, accuracy, loss, sim_score, p, a

    def vfinal_no_alpha(self):
        if self.config.task == 1:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 2:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 3:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 4:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 5:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        elif self.config.task == 6:
            self.g_theta_layers = [256, 128]
            self.f_phi_layers = [256, 512]
        else:
            log.error("Task index error")
        self.g_theta_layers.append(1)  # attention should be ended with layer sized 1
        md = Module(self.config, self.g_theta_layers, self.f_phi_layers)
        # embedded_c = md.contextLSTM(self.context, self.sentence_real_len, with_embed_matrix=True)
        # embedded_q = md.questionLSTM(self.question, self.question_real_len, with_embed_matrix=True)
        embedded_c = md.contextSum(self.context, with_embed_matrix=True)
        embedded_q = md.questionSum(self.question, with_embed_matrix=True)
        # embedded_c = md.add_label(embedded_c, self.speaker, self.order)
        m, alphas = md.hop_2(embedded_c, embedded_q, phase=self.is_training, norm='bn', activation=tf.nn.relu,
                             with_beta=True, with_alpha=False)
        input_ = md.concat_with_q(m[-1], embedded_q)
        pred = md.f_phi(input_, norm='bn', reuse=False, with_embed_matrix=True, phase=self.is_training)
        correct, accuracy, loss, sim_score, p, a = md.get_corr_acc_loss(pred, self.answer, self.answer_idx)
        self.alpha_1 = alphas[0]
        self.alpha_2 = alphas[1]
        return pred, correct, accuracy, loss, sim_score, p, a
