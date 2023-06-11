import tensorflow as tf
import tflearn
import numpy as np


class SelfAttentionNet(object):
    def __init__(self, sess, dim, optimizer, learning_rate, tau, grained, max_lenth, dropout, wordvector, logger):
        self.global_step = tf.Variable(0, trainable=False, name="CriticStep")
        self.sess = sess
        self.logger = logger
        self.max_lenth = max_lenth
        self.dim = dim
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.95, staircase=True)
        self.tau = tau
        self.grained = grained
        self.dropout = dropout
        self.init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.L2regular = 0.0001
        self.logger.info("optimizer: " + str(optimizer))
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.keep_prob = tf.placeholder(tf.float32, name="keepprob")
        self.num_other_variables = len(tf.trainable_variables())
        ############################
        self.wordvector = tf.get_variable('wordvector_active', dtype=tf.float32, initializer=wordvector, trainable=True)

        self.inputs, self.lenth, self.new_M, self.flatten_A, self.flatten_M, self.flatten_H, self.out2 = self.create_critic_atten(
            'Active')

        self.network_params = tf.trainable_variables()[self.num_other_variables:]
        ############################
        self.target_wordvector = tf.get_variable('wordvector_target', dtype=tf.float32, initializer=wordvector,
                                                 trainable=True)

        self.target_inputs, self.target_lenth, self.target_new_M, self.target_flatten_A, self.target_flatten_M, self.target_flatten_H, self.target_out2 = self.create_critic_atten(
            'Target')

        self.target_network_params = tf.trainable_variables()[len(self.network_params) + self.num_other_variables:]
        ############################
        self.update_target_network_params = \
            [self.target_network_params[i].assign( \
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1 - self.tau)) \
                for i in range(len(self.target_network_params))]

        self.assign_target_network_params = \
            [self.target_network_params[i].assign( \
                self.network_params[i]) for i in range(len(self.target_network_params))]

        self.assign_active_network_params = \
            [self.network_params[i].assign( \
                self.target_network_params[i]) for i in range(len(self.network_params))]
        ############################

        self.ground_truth = tf.placeholder(tf.float32, [1, self.grained], name="ground_truth")

        self.loss_target = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.target_out2)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.out2)

        self.loss2 = 0
        with tf.variable_scope("Active/pred2", reuse=True):
            self.loss2 += tf.nn.l2_loss(tf.get_variable('W'))

        self.loss += self.loss2 * self.L2regular
        self.loss_target += self.loss2 * self.L2regular
        self.gradients = tf.gradients(self.loss_target, self.target_network_params)
        self.logger.info(self.gradients)
        self.optimize = self.optimizer.apply_gradients(zip(self.gradients, self.network_params),
                                                       global_step=self.global_step)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        self.WVinput, self.WVvec = self.create_wordvector_find()

    def create_critic_atten(self, Scope):
        inputs = tf.placeholder(shape=[1, self.max_lenth], dtype=tf.int32, name="inputs")
        lenth = tf.placeholder(shape=[1], dtype=tf.int32, name="lenth")
        new_M = tf.placeholder(shape=[1, self.max_lenth, 2 * self.dim], dtype=tf.float32, name="new_M")

        if Scope[-1] == 'e':
            vec = tf.nn.embedding_lookup(self.wordvector, inputs)
        else:
            vec = tf.nn.embedding_lookup(self.target_wordvector, inputs)

        bw_cell = tf.contrib.rnn.LSTMCell(self.dim, initializer=self.init, state_is_tuple=False)
        fw_cell = tf.contrib.rnn.LSTMCell(self.dim, initializer=self.init, state_is_tuple=False)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(bw_cell, fw_cell, vec, lenth,
                                                                    dtype=tf.float32, scope=Scope + "/critic_atten")

        H = tf.concat([output_fw, output_bw], 2)
        with tf.variable_scope(Scope + "/attention", reuse=False):
            flatten_H = tf.transpose(H, [1, 0, 2])

            H_reshape = H[0, :, :]
            W_a = tf.get_variable("W_a", shape=[2 * self.dim, 2 * self.dim], initializer=self.initializer,
                                  trainable=True)
            H_a = tf.nn.tanh(tf.matmul(H_reshape, W_a))
            V_a = tf.get_variable("V_a", shape=[2 * self.dim, 1], initializer=self.initializer, trainable=True)
            H_a = tf.matmul(H_a, V_a)
            H_a = tf.expand_dims(tf.transpose(H_a, [1, 0]), 0)

            A = tf.transpose(tf.nn.softmax(H_a, name="attention"), [0, 2, 1])
            flatten_A = tf.reshape(A, [-1])

            M = H * A
            flatten_M = tf.transpose(M, [1, 0, 2])

            def package_new_M():
                return new_M

            def package_old_M():
                return M

            condition = tf.reduce_all(tf.equal(tf.zeros([1, self.max_lenth, 2 * self.dim], tf.float32), new_M))
            tmp_M = tf.cond(condition, true_fn=package_old_M, false_fn=package_new_M)

        # out1 = tf.reduce_sum(tmp_M, axis=2)
        out1 = tf.reshape(tmp_M, [-1, self.max_lenth * 2 * self.dim])
        out1 = tflearn.dropout(out1, self.keep_prob)
        out1_2 = tflearn.fully_connected(out1, 2 * self.dim, scope=Scope + "/pred1", name="get_pred")

        out2 = tflearn.dropout(out1_2, self.keep_prob)
        out2 = tflearn.fully_connected(out2, self.grained, scope=Scope + "/pred2", name="get_pred")

        return inputs, lenth, new_M, flatten_A, flatten_M, flatten_H, out2

    def create_wordvector_find(self):
        inputs = tf.placeholder(dtype=tf.int32, shape=[1, self.max_lenth], name="WVtofind")
        vec = tf.nn.embedding_lookup(self.target_wordvector, inputs)
        return inputs, vec

    def getloss_without_rl_att(self, lenth, inputs, ground_truth):

        return self.sess.run([self.target_out2, self.loss_target], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: np.zeros((1, self.max_lenth, 2 * self.dim), dtype=np.float32),
            self.ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def get_A_M_H(self, lenth, inputs):
        return self.sess.run([self.target_flatten_A, self.target_flatten_M, self.target_flatten_H], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: np.zeros((1, self.max_lenth, 2 * self.dim), dtype=np.float32),
        })

    def getloss_with_rl_att(self, lenth, inputs, target_flatten_m, ground_truth):

        return self.sess.run([self.target_out2, self.loss_target], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: target_flatten_m,
            self.ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def train_without_rl_att(self, lenth, inputs, ground_truth):

        return self.sess.run([self.target_out2, self.loss_target, self.optimize], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: np.zeros((1, self.max_lenth, 2 * self.dim), dtype=np.float32),
            self.ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def train_with_rl_att(self, lenth, inputs, target_flatten_m, ground_truth):
        return self.sess.run([self.target_out2, self.loss_target, self.optimize], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: target_flatten_m,
            self.ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def predict_target_without_rl_att(self, lenth, inputs):

        return self.sess.run([self.target_out2], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: np.zeros((1, self.max_lenth, 2 * self.dim), dtype=np.float32),
            self.keep_prob: 1.0})

    def predict_target_with_rl_att(self, lenth, inputs, target_flatten_m):
        return self.sess.run([self.target_out2], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.target_new_M: target_flatten_m,
            self.keep_prob: 1.0})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)

    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars, self.num_other_variables

    def wordvector_find(self, inputs):
        return self.sess.run(self.WVvec, feed_dict={
            self.WVinput: inputs})


class PolicyNet(object):
    def __init__(self, sess, maxlenth, dim, optimizer, learning_rate, tau, logger):
        self.global_step = tf.Variable(0, trainable=False, name="ActorStep")
        self.sess = sess
        self.maxlenth = maxlenth
        self.logger = logger
        self.dim = dim
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.95, staircase=True)
        self.tau = tau
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.num_other_variables = len(tf.trainable_variables())
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.input_summed, self.input_cur, self.input_wv, self.input_ori_sum, self.scaled_out = self.create_actor_network(
            'Actor')
        self.network_params = tf.trainable_variables()[self.num_other_variables:]

        self.target_input_summed, self.target_input_cur, self.target_input_wv, self.target_input_ori_sum, self.target_scaled_out = self.create_actor_network(
            'Target')
        self.target_network_params = tf.trainable_variables()[self.num_other_variables + len(self.network_params):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign( \
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1 - self.tau)) \
                for i in range(len(self.target_network_params))]

        self.assign_active_network_params = \
            [self.network_params[i].assign( \
                self.target_network_params[i]) for i in range(len(self.network_params))]

        self.action_gradient = tf.placeholder(tf.float32, [2])  # reward
        self.log_target_scaled_out = tf.log(self.target_scaled_out)

        self.actor_gradients = tf.gradients(self.log_target_scaled_out, self.target_network_params,
                                            self.action_gradient)

        self.grads = [tf.placeholder(tf.float32, [2 * self.dim, 2]),
                      tf.placeholder(tf.float32, [2 * self.dim, 2]),
                      tf.placeholder(tf.float32, [self.dim, 2]),
                      tf.placeholder(tf.float32, [2 * self.dim, 2]),
                      tf.placeholder(tf.float32, [2, ])
                      ]
        self.optimize = self.optimizer.apply_gradients(zip(self.grads, self.network_params),
                                                       global_step=self.global_step)

    def create_actor_network(self, Scope):
        input_attented_vec = tf.placeholder(tf.float32, shape=[1, 2 * self.dim])  # self.maxlenth,
        input_cur = tf.placeholder(tf.float32, shape=[1, 2 * self.dim])
        input_wv = tf.placeholder(tf.float32, shape=[1, self.dim])
        input_ori_sum = tf.placeholder(tf.float32, shape=[1, 2 * self.dim])

        with tf.variable_scope(Scope, reuse=False):
            W_1 = tf.get_variable("W_1", shape=[2 * self.dim, 2], initializer=self.initializer)
            W_2 = tf.get_variable("W_2", shape=[2 * self.dim, 2], initializer=self.initializer)
            W_3 = tf.get_variable("W_3", shape=[self.dim, 2], initializer=self.initializer)
            W_4 = tf.get_variable("W_4", shape=[2 * self.dim, 2], initializer=self.initializer)
            b = tf.get_variable("b", shape=[2, ], initializer=self.initializer)

            scaled_out = tflearn.activation(
                tf.matmul(input_attented_vec, W_1) + tf.matmul(input_cur, W_2)
                + tf.matmul(input_wv, W_3) + tf.matmul(input_ori_sum, W_4)
                + b,
                activation='softmax')
        return input_attented_vec, input_cur, input_wv, input_ori_sum, scaled_out[0]

    def train(self, grad):
        self.sess.run(self.optimize, feed_dict={
            self.grads[0]: grad[0],
            self.grads[1]: grad[1],
            self.grads[2]: grad[2],
            self.grads[3]: grad[3],
            self.grads[4]: grad[4],
        })

    def predict_target(self, input_summed, input_cur, input_wv, input_ori_sum):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_input_summed: input_summed,
            self.target_input_cur: input_cur,
            self.target_input_wv: input_wv,
            self.target_input_ori_sum: input_ori_sum,
        })

    def get_gradient(self, input_summed, input_cur, input_wv, input_ori_sum, a_gradient):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.target_input_summed: input_summed,
            self.target_input_cur: input_cur,
            self.target_input_wv: input_wv,
            self.target_input_ori_sum: input_ori_sum,
            self.action_gradient: a_gradient})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)
