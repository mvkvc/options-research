import os, sys
lib_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(lib_path)

import tensorflow as tf
from ddpg_daibing.ActorNetwork import DDPG_Actor
from ddpg_daibing.CriticNetwork  import DDPG_Critic
import numpy as np

class Model(object):
    def __init__(self,
                 type,
                 state_dim=8,
                 action_dim=1,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3,
                 tau = 0.001,
                 before_train=0,
                 sess=None):
        self.before_train=before_train
        self.type=type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau

        #tf.reset_default_graph()
        self.sess = sess or tf.Session()

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        global_step_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
        self.sess.run(tf.variables_initializer(global_step_vars))

        self.actor_scope = "actor_net"
        with tf.name_scope(self.actor_scope):
            self.actor = DDPG_Actor(
                        self.type,
                        self.state_dim,
                        self.action_dim,
                        learning_rate=self.actor_learning_rate,
                        tau=self.tau,
                        scope=self.actor_scope,
                        sess=self.sess)

        self.critic_scope = "critic_net"
        with tf.name_scope(self.critic_scope):
            self.critic = DDPG_Critic(self.state_dim,
                        self.action_dim,
                        learning_rate=self.critic_learning_rate,
                        tau=self.tau,
                        scope=self.critic_scope,
                        sess=self.sess)

    def update(self, state_batch, action_batch, y_batch, sess=None):
        sess = sess or self.sess
        self.critic.update_source_critic_net(state_batch, action_batch, y_batch, sess)
        action_batch_for_grad = self.actor.predict_action_source_net(state_batch, sess)
        action_grad_batch = self.critic.get_action_grads(state_batch, action_batch_for_grad, sess)
        self.actor.update_source_actor_net(state_batch, action_grad_batch, sess)
        if self.before_train==1:
            y_=tf.placeholder(tf.float32,[None,1],name='y-input')
            losss=tf.losses.mean_squared_error(y_,self.actor.action_output)
            train_add = self.actor.optimizer.minimize(losss)
            t=np.reshape(state_batch[:, 5],(len(state_batch),1))
            init_opo=tf.global_variables_initializer()
            sess.run(init_opo)
            sess.run(train_add,feed_dict={
                y_:t,self.actor.input_state:state_batch
            })
        self.critic.update_target_critic_net(sess)
        self.actor.update_target_actor_net(sess)

    def predict_action(self, observation, sess=None):
        sess = sess or self.sess
        return self.actor.predict_action_source_net(observation, sess)

if __name__ == '__main__':
   pass