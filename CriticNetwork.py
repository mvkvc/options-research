import tensorflow as tf
from math import sqrt

class DDPG_Critic(object):
    def __init__(self, state_dim, action_dim, optimizer=None, learning_rate=0.001, tau=0.001, scope="", sess=None):
        self.scope = scope
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.l2_reg = 0.01
        self.optimizer = optimizer or tf.train.AdamOptimizer(self.learning_rate)
        self.tau = tau
        self.h1_dim =2400 #400
        self.h2_dim =3600 #600
        self.h3_dim =2400 #400
        self.loss_critic=[]
        self.activation = tf.nn.relu
        self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        # fan-out uniform initializer which is different from original paper
        self.kernel_initializer_1 = tf.random_uniform_initializer(minval=-1/sqrt(self.h1_dim), maxval=1/sqrt(self.h1_dim))
        self.kernel_initializer_2 = tf.random_uniform_initializer(minval=-1/sqrt(self.h2_dim), maxval=1/sqrt(self.h2_dim))
        self.kernel_initializer_3 = tf.random_uniform_initializer(minval=-1/sqrt(self.h3_dim), maxval=1/sqrt(self.h3_dim))
        self.kernel_initializer_4 = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        # self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
        self.kernel_regularizer=None

        with tf.name_scope("critic_input"):
            self.input_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="states")
            self.input_action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="actions")

        with tf.name_scope("critic_label"):
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name="y")

        self.source_var_scope = "ddpg/" + "critic_net"
        with tf.variable_scope(self.source_var_scope):
            self.q_output = self.__create_critic_network()

        self.target_var_scope = "ddpg/" + "critic_target_net"
        with tf.variable_scope(self.target_var_scope):
            self.target_net_q_output = self.__create_target_network()

        with tf.name_scope("compute_critic_loss"):
            self.__create_loss()

        self.train_op_scope = "critic_train_op"
        with tf.variable_scope(self.train_op_scope):
            self.__create_train_op()

        with tf.name_scope("critic_target_update_train_op"):
            self.__create_update_target_net_op()

        with tf.name_scope("get_action_grad_op"):
            self.__create_get_action_grad_op()

        self.__create_get_layer_weight_op_source()
        self.__create_get_layer_weight_op_target()

    def __create_critic_network(self):
        h1 = tf.layers.dense(self.input_state,
                                units=self.h1_dim,
                                activation=self.activation,
                                kernel_initializer=self.kernel_initializer_1,
                                kernel_regularizer=self.kernel_regularizer,
                                name="hidden_1")

        h2 = tf.layers.dense(self.input_action,
                                units=self.h2_dim,
                                activation=self.activation,
                                kernel_initializer=self.kernel_initializer_2,
                                kernel_regularizer=self.kernel_regularizer,
                                name="hidden_2")

        h_concat = tf.concat([h1, h2], 1, name="h_concat")

        h3 = tf.layers.dense(h_concat,
                                units=self.h3_dim,
                                activation=self.activation,
                                kernel_initializer=self.kernel_initializer_3,
                                kernel_regularizer=self.kernel_regularizer,
                                name="hidden_3")


        q_output = tf.layers.dense(h3,
                                units=1,
                                activation=None,
                                # activation =self.activation,
                                kernel_initializer=self.kernel_initializer_4,
                                kernel_regularizer=self.kernel_regularizer,
                                name="q_output")

        return q_output

    def __create_target_network(self):
        # get source variales and initialize
        source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.source_var_scope)
        self.sess.run(tf.variables_initializer(source_vars))

        # create target network and initialize it by source network
        q_output = self.__create_critic_network()
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_var_scope)

        target_init_op_list = [target_vars[i].assign(source_vars[i]) for i in range(len(source_vars))]
        self.sess.run(target_init_op_list)

        return q_output

    def __create_loss(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.q_output)

    def __create_train_op(self):
        self.train_q_op = self.optimizer.minimize(self.loss)
        train_op_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.scope + "/" + self.train_op_scope) # to do: remove prefix
        train_op_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.train_op_scope))
        self.sess.run(tf.variables_initializer(train_op_vars))

    def __create_update_target_net_op(self):
        source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.source_var_scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_var_scope)
        update_target_net_op_list = [target_vars[i].assign(self.tau*source_vars[i] + (1-self.tau)*target_vars[i]) for i in range(len(source_vars))]
        self.update_target_net_op = tf.group(*update_target_net_op_list)

    def __create_get_action_grad_op(self):
        self.get_action_grad_op = tf.gradients(self.q_output, self.input_action)

    def predict_q_source_net(self, feed_state, feed_action, sess=None):
        sess = sess or self.sess
        return sess.run(self.q_output, {self.input_state: feed_state,
                                        self.input_action: feed_action})

    def predict_q_target_net(self, feed_state, feed_action, sess=None):
        sess = sess or self.sess
        return sess.run(self.target_net_q_output, {self.input_state: feed_state,
                                             self.input_action: feed_action})

    def update_source_critic_net(self, feed_state, feed_action, feed_y, sess=None):
        sess = sess or self.sess
        loss_=sess.run([self.loss],
                        {self.input_state: feed_state,
                         self.input_action: feed_action,
                         self.y: feed_y})
        self.loss_critic.append(loss_)
        return sess.run([self.train_q_op],
                        {self.input_state: feed_state,
                         self.input_action: feed_action,
                         self.y: feed_y})

    def update_target_critic_net(self, sess=None):
        sess = sess or self.sess
        return sess.run(self.update_target_net_op)

    def get_action_grads(self, feed_state, feed_action, sess=None):
        sess = sess or self.sess
        return (sess.run(self.get_action_grad_op, {self.input_state: feed_state,
                                                  self.input_action: feed_action}))[0]

    def __create_get_layer_weight_op_source(self):
        with tf.variable_scope(self.source_var_scope, reuse=True):
            self.h1_weight_source = tf.get_variable("hidden_1/kernel")
            self.h1_bias_source = tf.get_variable("hidden_1/bias")

    def run_layer_weight_source(self, sess=None):
        sess = sess or self.sess
        return sess.run([self.h1_weight_source, self.h1_bias_source])

    def __create_get_layer_weight_op_target(self):
        with tf.variable_scope(self.target_var_scope, reuse=True):
            self.h1_weight_target = tf.get_variable("hidden_1/kernel")
            self.h1_bias_target = tf.get_variable("hidden_1/bias")

    def run_layer_weight_target(self, sess=None):
        sess = sess or self.sess
        return sess.run([self.h1_weight_target, self.h1_bias_target])

if __name__ == '__main__':
           pass
