#-*-coding:utf-8-*-
from keras.layers import Dense, Input, RepeatVector, Lambda, Permute, Multiply, Concatenate, Add
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.engine.topology import Layer

from keras import backend as K
from keras.layers.wrappers import TimeDistributed
import numpy as np
np.random.seed(1337)

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


class _WordAttention(object):
    def __init__(self):
        pass

    def __call__(self, sen1, sen2):
        def _outer(AB):
            return K.batch_dot(AB[0], K.permute_dimensions(AB[1], (0, 2, 1)))

        len_x = K.int_shape(sen1)[1] # (batch, enc(dec), hidden_dim)
        len_y = K.int_shape(sen2)[1] # (batch, dec(enc), hidden_dim)
        return Lambda(_outer, output_shape=(len_x, len_y,))([sen1, sen2])


class _EntAttention(Layer):
    def __init__(self, entity_dim, **kwargs):
        self.entity_dim = entity_dim
        super(_EntAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.kernel = self.add_weight(name='ent_sim_weight',
                                    shape=(self.entity_dim, self.entity_dim),
                                    initializer='uniform',
                                    trainable=True)
        super(_EntAttention, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        s_e_1, s_e_2 = x  # [(batch, enc(dec), ent_dim), (batch, dec(enc), ent_dim)]
        weight_s_e_2 = K.dot(s_e_2, self.kernel)  # (batch, dec(enc), ent_dim)
        return K.batch_dot(s_e_1, K.permute_dimensions(weight_s_e_2,(0, 2, 1)))  # (batch, enc(dec), dec(enc))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_a[1], shape_b[1])


class _EntAttMean(Layer):
    def __init__(self, hidden_dim, entity_dim, **kwargs):
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.att_dim = 200
        super(_EntAttMean, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.We = self.add_weight(name='We',
                                    shape=(self.entity_dim, self.att_dim),
                                    initializer='uniform',
                                    trainable=True)
        self.Wh = self.add_weight(name='Wh',
                                    shape=(self.hidden_dim, self.att_dim),
                                    initializer='uniform',
                                    trainable=True)
        self.Ws = self.add_weight(name='Ws',
                                    shape=(self.att_dim, 1),
                                    initializer='uniform',
                                    trainable=True)
        super(_EntAttMean, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        cxt, ent = x  # [(batch, enc(dec), hidden_dim), (batch, enc(dec), n_entity, ent_dim)]
        seq_len = K.int_shape(cxt)[1]
        n_entity = K.int_shape(ent)[2]

        reshape_cxt = tf.reduce_mean(cxt, axis=1)  # (batch, hidden_dim)
        reshape_cxt = tf.expand_dims(reshape_cxt, axis=1)  # (batch, 1, hidden_dim)
        reshape_cxt = tf.expand_dims(reshape_cxt, axis=1)  # (batch, 1, 1, hidden_dim)
        reshape_cxt = tf.tile(reshape_cxt, [1, seq_len, n_entity, 1])  # (batch, enc(dec), n_entity, hidden_dim)
        reshape_cxt = tf.reshape(reshape_cxt, [-1, self.hidden_dim])  # (batch * enc(dec) * n_entity, hidden_dim)

        reshape_ent = tf.reshape(ent, [-1, self.entity_dim])  # (batch * enc(dec) * n_entity, entity_dim)

        M = tf.tanh(tf.add(tf.matmul(reshape_cxt, self.Wh), tf.matmul(reshape_ent, self.We)))  # (batch * enc(dec) * n_entity, attention_dim)
        M = tf.matmul(M, self.Ws)  # (batch * enc(dec) * n_entity, 1)

        S = tf.reshape(M, [-1, n_entity])  # (batch * enc(dec), n_entity)
        S = tf.nn.softmax(S)  # (batch * enc(dec), n_entity)

        S_diag = tf.matrix_diag(S)  # (batch * enc(dec), n_entity, n_entity)
        reshape_ent = tf.reshape(ent, [-1, n_entity, self.entity_dim])  # (batch * enc(dec), n_entity, ent_dim)
        ent_att = tf.matmul(S_diag, reshape_ent)  # (batch * enc(dec), n_entity, ent_dim)
        ent_att = tf.reshape(ent_att, [-1, seq_len, n_entity, self.entity_dim])  # (batch, enc(dec), n_entity, ent_dim)
        ent_att = tf.reduce_sum(ent_att, axis=2)  # (batch, enc(dec), ent_dim)

        return ent_att

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_a[1], shape_b[3])


class _WordSoftAlignment(object):
    def __init__(self, k_value_que, k_value_ans, model_type="listwise"):
        self.k_value_que = k_value_que
        self.k_value_ans = k_value_ans
        self.model_type = model_type

    def __call__(self, attention, sentence, sen_length, reverse=False):
        def _normalize_attention(attmat):
            att = attmat[0] # (batch, enc(dec), dec(enc))
            sen = attmat[1] # (batch, dec(enc), hidden_dim)
            sen_len = attmat[2]  # (batch, dec(enc))
            num = K.int_shape(att)[1]

            reshape_len = K.expand_dims(sen_len, axis=1) # (batch, 1, dec(enc))
            reshape_len = K.tile(reshape_len, [1, num, 1]) # (batch, enc(dec), dec(enc))

            e = K.softmax(att)  # (batch, enc(dec), dec(enc))
            g = e * reshape_len  # (batch, enc(dec), dec(enc))

            if self.model_type == "k_max":
                if reverse:
                    bound = int(self.k_value_que)
                else:
                    bound = int(self.k_value_ans)
                top_val, _ = tf.math.top_k(g, k=bound, sorted=True)
                low_bound = tf.reduce_min(top_val, axis=-1, keepdims=True)
                g = K.switch(g < low_bound, tf.zeros_like(g), g)

            elif self.model_type == "k_threhold":
                if reverse:
                    if self.k_value_que == -1:
                        threshold = 1.0 / K.sum(sen_len, axis=-1, keepdims=True)
                    else:
                        threshold = self.k_value_que
                else:
                    if self.k_value_ans == -1:
                        threshold = 1.0/K.sum(sen_len, axis=-1, keepdims=True)
                    else:
                        threshold = self.k_value_ans
                k_threhold_s = K.clip(K.sum(g, axis=-1, keepdims=True), 0.00001, 1.0)  # (batch, enc(dec), 1)
                k_threhold_att = g / k_threhold_s  # (batch, enc(dec), dec(enc))
                g = K.switch(k_threhold_att < threshold, tf.zeros_like(k_threhold_att), k_threhold_att)
            else:
                pass

            s = K.clip(K.sum(g, axis=-1, keepdims=True), 0.00001, 1.0)  # (batch, enc(dec), 1)
            sm_att = g / s  # (batch, enc(dec), dec(enc))
            return K.batch_dot(sm_att, sen)  # (batch, enc(dec), hidden_dim)

        max_length = K.int_shape(attention)[1] # enc(dec)
        hidden_dim = K.int_shape(sentence)[2] # hidden_dim
        return Lambda(_normalize_attention, output_shape=(max_length, hidden_dim))([attention, sentence, sen_length])


class _EntSoftAlignment(object):
    def __init__(self, k_value_que, k_value_ans, model_type="listwise"):
        self.k_value_que = k_value_que
        self.k_value_ans = k_value_ans
        self.model_type = model_type

    def __call__(self, attention, sentence_entity, sen_length, reverse=False):
        def _normalize_attention(attmat):
            att = attmat[0] # (batch, enc(dec), dec(enc))
            sen_ent = attmat[1] # (batch, dec(enc), ent_dim)
            sen_len = attmat[2]  # (batch, dec(enc))
            num = K.int_shape(att)[1]

            reshape_len = K.expand_dims(sen_len, axis=1) # (batch, 1, dec(enc))
            reshape_len = K.tile(reshape_len, [1, num, 1]) # (batch, enc(dec), dec(enc))

            e = K.softmax(att)  # (batch, enc(dec), dec(enc))
            g = e * reshape_len  # (batch, enc(dec), dec(enc))

            if self.model_type == "k_max":
                if reverse:
                    bound = int(self.k_value_que)
                else:
                    bound = int(self.k_value_ans)
                top_val, _ = tf.math.top_k(g, k=bound, sorted=True)
                low_bound = tf.reduce_min(top_val, axis=-1, keepdims=True)  # (batch, enc(dec), 1)
                g = K.switch(g < low_bound, tf.zeros_like(g), g)  # (batch, dec(enc))

            elif self.model_type == "k_threhold":
                if reverse:
                    if self.k_value_que == -1:
                        threshold = 1.0 / K.sum(sen_len, axis=-1, keepdims=True)
                    else:
                        threshold = self.k_value_que
                else:
                    if self.k_value_ans == -1:
                        threshold = 1.0 / K.sum(sen_len, axis=-1, keepdims=True)
                    else:
                        threshold = self.k_value_ans
                k_threhold_s = K.clip(K.sum(g, axis=-1, keepdims=True), 0.00001, 1.0)  # (batch, 1)
                k_threhold_att = g / k_threhold_s  # (batch, dec(enc))
                g = K.switch(k_threhold_att < threshold, tf.zeros_like(k_threhold_att), k_threhold_att)  # (batch, dec(enc))
            else:
                pass

            s = K.clip(K.sum(g, axis=-1, keepdims=True), 0.00001, 1.0)  # (batch, enc(dec), 1)
            sm_att = g / s  # (batch, enc(dec), dec(enc))
            return K.batch_dot(sm_att, sen_ent)

        max_length = K.int_shape(attention)[1] # enc(dec)
        entity_dim = K.int_shape(sentence_entity)[2] # ent_dim
        return Lambda(_normalize_attention, output_shape=(max_length, entity_dim,))([attention, sentence_entity, sen_length])


class ModelFactory(object):
    @staticmethod
    def get_basic_model(model_param, embedding_file, ent_embedding_file, WordAttend, EntAttMean, EntAttend, WordSoftAlign, EntSoftAlign, cxt_weight_mixed=True, ent_weight_mixed=True, entity_compare=True):
        hidden_dim = model_param.hidden_dim
        entity_dim = model_param.entity_dim
        n_entity = model_param.n_entity

        question = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')
        question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')
        question_ent = Input(shape=(model_param.enc_timesteps, model_param.n_entity,), dtype='float32', name='question_ent')

        answer = Input(shape=(model_param.dec_timesteps,), dtype='float32', name='answer_base')
        answer_len = Input(shape=(model_param.dec_timesteps,), dtype='float32', name='answer_len')
        answer_ent = Input(shape=(model_param.dec_timesteps, model_param.n_entity,), dtype='float32', name='answer_ent')

        # =====================================================================================================
        # ===================================== Word Representation Layer =====================================
        # =====================================================================================================
        weights = np.load(embedding_file)
        weights[0] = np.zeros((weights.shape[1]))

        QaEmbedding = Embedding(input_dim=weights.shape[0],
                                output_dim=weights.shape[1],
                                weights=[weights],
                                trainable=False)

        question_emb = QaEmbedding(question)
        answer_emb = QaEmbedding(answer)

        SigmoidDense = Dense(hidden_dim, activation="sigmoid")
        TanhDense = Dense(hidden_dim, activation="tanh")

        QueTimeSigmoidDense = TimeDistributed(SigmoidDense, name="que_time_s")
        QueTimeTanhDense = TimeDistributed(TanhDense, name="que_time_t")
        question_sig = QueTimeSigmoidDense(question_emb)
        question_tanh = QueTimeTanhDense(question_emb)
        question_proj = Multiply()([question_sig, question_tanh])  # (batch_size, enc, hidden_dim)

        AnsTimeSigmoidDense = TimeDistributed(SigmoidDense, name="ans_time_s")
        AnsTimeTanhDense = TimeDistributed(TanhDense, name="ans_time_t")
        answer_sig = AnsTimeSigmoidDense(answer_emb)
        answer_tanh = AnsTimeTanhDense(answer_emb)
        answer_proj = Multiply()([answer_sig, answer_tanh]) # (batch_size, dec, hidden_dim)

        # =====================================================================================================
        # ======================================== Word Attention Layer =======================================
        # =====================================================================================================
        que_atten_metrics = WordAttend(question_proj, answer_proj) # (batch_size, enc, dec)
        ans_atten_metrics = WordAttend(answer_proj, question_proj) # (batch_size, dec, enc)

        # =======================================================================================================
        # ===================================== Entity Representation Layer =====================================
        # =======================================================================================================
        ent_vec = np.load(ent_embedding_file)
        EntEmbedding = Embedding(input_dim=ent_vec.shape[0],
                                output_dim=ent_vec.shape[1],
                                weights=[ent_vec],
                                trainable=False)

        question_ent_emb = EntEmbedding(question_ent) # (batch, enc, n_entity, ent_dim)
        answer_ent_emb = EntEmbedding(answer_ent) # (batch, dec, n_entity, ent_dim)

        question_ent_emb_att = EntAttMean([question_proj, question_ent_emb])  # (batch, enc, ent_dim)
        answer_ent_emb_att = EntAttMean([answer_proj, answer_ent_emb])  # (batch, dec, ent_dim)

        EntSigmoidDense = Dense(entity_dim, activation="sigmoid")
        EntTanhDense = Dense(entity_dim, activation="tanh")

        EntQueTimeSigmoidDense = TimeDistributed(EntSigmoidDense, name="ent_que_time_s")
        EntQueTimeTanhDense = TimeDistributed(EntTanhDense, name="ent_que_time_t")
        ent_question_sig = EntQueTimeSigmoidDense(question_ent_emb_att)
        ent_question_tanh = EntQueTimeTanhDense(question_ent_emb_att)
        question_ent_emb_att = Multiply()([ent_question_sig, ent_question_tanh])  # (batch_size, enc, hidden_dim)

        EntAnsTimeSigmoidDense = TimeDistributed(EntSigmoidDense, name="ent_ans_time_s")
        EntAnsTimeTanhDense = TimeDistributed(EntTanhDense, name="ent_ans_time_t")
        ent_answer_sig = EntAnsTimeSigmoidDense(answer_ent_emb_att)
        ent_answer_tanh = EntAnsTimeTanhDense(answer_ent_emb_att)
        answer_ent_emb_att = Multiply()([ent_answer_sig, ent_answer_tanh]) # (batch_size, dec, hidden_dim)

        # =======================================================================================================
        # ======================================= Entity Attention Layer ========================================
        # =======================================================================================================
        que_ent_atten_metrics = EntAttend([question_ent_emb_att, answer_ent_emb_att])  # (batch, enc, dec)
        ans_ent_atten_metrics = EntAttend([answer_ent_emb_att, question_ent_emb_att])  # (batch, dec, enc)

        # =======================================================================================================
        # ========================================== Attention Merge ============================================
        # =======================================================================================================
        temp_que_atten_metrics = Lambda(lambda x:K.softmax(x[0], axis=-1) + K.softmax(x[1], axis=-1), output_shape=(model_param.enc_timesteps, model_param.dec_timesteps))([que_atten_metrics, que_ent_atten_metrics])
        temp_ans_atten_metrics = Lambda(lambda x:K.softmax(x[0], axis=-1) + K.softmax(x[1], axis=-1), output_shape=(model_param.dec_timesteps, model_param.enc_timesteps))([ans_atten_metrics, ans_ent_atten_metrics])

        if cxt_weight_mixed:
            que_atten_metrics = temp_que_atten_metrics
            ans_atten_metrics = temp_ans_atten_metrics

        if ent_weight_mixed:
            que_ent_atten_metrics = temp_que_atten_metrics
            ans_ent_atten_metrics = temp_ans_atten_metrics

        # =======================================================================================================
        # ======================================= Word Comparison Layer =========================================
        # =======================================================================================================
        question_align = WordSoftAlign(que_atten_metrics, answer_proj, answer_len) # (batch, enc, hidden)
        answer_align = WordSoftAlign(ans_atten_metrics, question_proj, question_len, reverse=True) # (batch, dec, hidden)

        que_temp_sim_output = Multiply()([question_proj, question_align])  # (batch, enc, hidden)
        ans_temp_sim_output = Multiply()([answer_proj, answer_align])  # (batch, dec, hidden)

        que_repeat_len = RepeatVector(hidden_dim)(question_len)  # (batch, hidden, enc)
        que_repear_vec = Permute((2,1))(que_repeat_len)  # (batch, enc, hidden)

        ans_repeat_len = RepeatVector(hidden_dim)(answer_len) # (batch, hidden, dec)
        ans_repear_vec = Permute((2,1))(ans_repeat_len)  # (batch, dec, hidden)

        que_sim_output = Multiply()([que_temp_sim_output,que_repear_vec])  # (batch, enc, hidden)
        ans_sim_output = Multiply()([ans_temp_sim_output,ans_repear_vec])  # (batch, dec, hidden)

        # =======================================================================================================
        # ===================================== Word Aggregation Layer ==========================================
        # =======================================================================================================
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

        cnns_q = [Conv1D(kernel_size=filter_length,
                            filters=hidden_dim,
                            activation='relu',
                            padding='same') for filter_length in [1,2,3,4,5]]

        cnn_feature_q = Concatenate()([cnn(que_sim_output) for cnn in cnns_q])
        cnn_pool_q = maxpool(cnn_feature_q) # (batch, hidden_dim * 5)
        OutputDense_q = Dense(hidden_dim, activation="tanh")
        feature_q = OutputDense_q(cnn_pool_q) # (batch, hidden_dim)


        cnns_a = [Conv1D(kernel_size=filter_length,
                            filters=hidden_dim,
                            activation='relu',
                            padding='same') for filter_length in [1,2,3,4,5]]

        cnn_feature_a = Concatenate()([cnn(ans_sim_output) for cnn in cnns_a])
        cnn_pool_a = maxpool(cnn_feature_a)
        OutputDense_a = Dense(hidden_dim, activation="tanh")
        feature_a = OutputDense_a(cnn_pool_a)

        # =======================================================================================================
        # ====================================== Enttity Comparison Layer =======================================
        # =======================================================================================================
        question_ent_align = EntSoftAlign(que_ent_atten_metrics, answer_ent_emb_att, answer_len) # (batch, enc, ent_dim)
        answer_ent_align = EntSoftAlign(ans_ent_atten_metrics, question_ent_emb_att, question_len, reverse=True) # (batch, dec, ent_dim)

        que_temp_cmp_ent_output = Multiply()([question_ent_emb_att, question_ent_align])  # (batch, enc, ent_dim)
        ans_temp_cmp_ent_output = Multiply()([answer_ent_emb_att, answer_ent_align])  # (batch, dec, ent_dim)

        que_repeat_len_ent = RepeatVector(entity_dim)(question_len)  # (batch, ent_dim, enc)
        que_repear_vec_ent = Permute((2,1))(que_repeat_len_ent)  # (batch, enc, ent_dim)

        ans_repeat_len_ent = RepeatVector(entity_dim)(answer_len) # (batch, ent_dim, dec)
        ans_repear_vec_ent = Permute((2,1))(ans_repeat_len_ent)  # (batch, dec, ent_dim)

        que_cmp_ent_output = Multiply()([que_temp_cmp_ent_output, que_repear_vec_ent])  # (batch, enc, ent_dim)
        ans_cmp_ent_output = Multiply()([ans_temp_cmp_ent_output, ans_repear_vec_ent])  # (batch, dec, ent_dim)

        # =======================================================================================================
        # ====================================== Enttity Aggregation Layer =======================================
        # =======================================================================================================
        maxpool_ent = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

        cnns_q_ent = [Conv1D(kernel_size=filter_length,
                            filters=entity_dim,
                            activation='relu',
                            padding='same') for filter_length in [1,2,3,4,5]]

        cnn_feature_q_ent = Concatenate()([cnn(que_cmp_ent_output) for cnn in cnns_q_ent])
        cnn_pool_q_ent = maxpool_ent(cnn_feature_q_ent) # (batch, ent_dim * 5)
        OutputDense_q_ent = Dense(hidden_dim, activation="tanh")
        feature_q_ent = OutputDense_q_ent(cnn_pool_q_ent) # (batch, hidden_dim)

        cnns_a_ent = [Conv1D(kernel_size=filter_length,
                            filters=entity_dim,
                            activation='relu',
                            padding='same') for filter_length in [1,2,3,4,5]]

        cnn_feature_a_ent = Concatenate()([cnn(ans_cmp_ent_output) for cnn in cnns_a_ent])
        cnn_pool_a_ent = maxpool_ent(cnn_feature_a_ent)
        OutputDense_a_ent = Dense(hidden_dim, activation="tanh")
        feature_a_ent = OutputDense_a_ent(cnn_pool_a_ent)  # (batch, hidden_dim)

        # =======================================================================================================
        # ========================================= Final Score Layer ===========================================
        # =======================================================================================================
        if entity_compare:
            feature_total = Concatenate()([feature_q, feature_q_ent, feature_a, feature_a_ent])  # (batch, hidden_dim * 4)
        else:
            feature_total = Concatenate()([feature_q, feature_a]) # (batch, hidden_dim * 2)

        FinalDense = Dense(hidden_dim, activation="tanh")
        feature_all = FinalDense(feature_total)  # (batch, hidden_dim)

        ScoreDense = Dense(1)
        score = ScoreDense(feature_all)

        model = Model(inputs=[question, answer, question_len, answer_len, question_ent, answer_ent], outputs=[score])
        return model

    @staticmethod
    def get_listwise_model(model_param, embedding_file, ent_embedding_file, WordAttend, EntAttMean, EntAttend, WordSoftAlign, EntSoftAlign, cxt_weight_mixed=True, ent_weight_mixed=True, entity_compare=True):
        basic_model = ModelFactory.get_basic_model(model_param, embedding_file, ent_embedding_file, WordAttend, EntAttMean, EntAttend, WordSoftAlign, EntSoftAlign, cxt_weight_mixed, ent_weight_mixed, entity_compare)

        question = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')
        question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')
        question_ent = Input(shape=(model_param.enc_timesteps, model_param.n_entity,), dtype='float32', name='question_ent')

        # For Predict
        single_answer = Input(shape=(model_param.dec_timesteps,), dtype='float32', name='single_answer_base')
        single_answer_len = Input(shape=(model_param.dec_timesteps,),dtype='float32', name='single_answer_len')
        single_answer_ent = Input(shape=(model_param.dec_timesteps, model_param.n_entity,), dtype='float32', name='single_answer_ent')

        single_similarity = basic_model([question, single_answer, question_len, single_answer_len, question_ent, single_answer_ent])

        # For Train
        answers = Input(shape=(model_param.random_size, model_param.dec_timesteps,), dtype='float32', name='answers_base')
        answers_len = Input(shape=(model_param.random_size, model_param.dec_timesteps,), dtype='float32', name='answers_len')
        answers_ent = Input(shape=(model_param.random_size, model_param.dec_timesteps, model_param.n_entity,), dtype='float32', name='answers_ent')

        sim_list = []
        for i in range(model_param.random_size):
            convert_layer = Lambda(lambda x:x[:,i], output_shape=(model_param.dec_timesteps,))
            temp_tensor = convert_layer(answers)
            temp_length = convert_layer(answers_len)
            convert_layer_att = Lambda(lambda x:x[:,i],output_shape=(model_param.dec_timesteps,model_param.n_entity,))
            temp_ent = convert_layer_att(answers_ent)
            temp_sim = basic_model([question, temp_tensor, question_len, temp_length, question_ent, temp_ent])
            sim_list.append(temp_sim)
        total_sim = Concatenate()(sim_list)
        total_prob = Lambda(lambda x: K.log(K.softmax(x)), output_shape = (model_param.random_size, ))(total_sim)


        prediction_model = Model(
            inputs=[question, single_answer, question_len, single_answer_len, question_ent, single_answer_ent], outputs=single_similarity, name='prediction_model')
        prediction_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

        training_model = Model(
            inputs=[question, answers, question_len, answers_len, question_ent, answers_ent], outputs=total_prob, name='training_model')
        training_model.compile(
            loss=lambda y_true, y_pred: K.mean(y_true*(K.log(K.clip(y_true,0.00001,1)) - y_pred )) , optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        
        return training_model, prediction_model

    @staticmethod
    def get_model(model_param, embedding_file, ent_embedding_file, model_type="listwise", cxt_weight_mixed=True, ent_weight_mixed=True, entity_compare=True):
        hidden_dim = model_param.hidden_dim
        entity_dim = model_param.entity_dim
        k_value_que = model_param.k_value_que
        k_value_ans = model_param.k_value_ans

        print("cxt_weight_mixed :{}".format(cxt_weight_mixed))
        print("ent_weight_mixed: {}".format(ent_weight_mixed))
        print("entity_compare: {}".format(entity_compare))
        WordAttend = _WordAttention()
        EntAttend = _EntAttention(entity_dim)
        EntAttMean = _EntAttMean(hidden_dim, entity_dim)
        WordSoftAlign = _WordSoftAlignment(k_value_que, k_value_ans, model_type=model_type)
        EntSoftAlign = _EntSoftAlignment(k_value_que, k_value_ans, model_type=model_type)

        return ModelFactory.get_listwise_model(model_param, embedding_file, ent_embedding_file, WordAttend, EntAttMean, EntAttend, WordSoftAlign, EntSoftAlign, cxt_weight_mixed=cxt_weight_mixed, ent_weight_mixed=ent_weight_mixed, entity_compare=entity_compare)
