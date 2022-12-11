import numpy as np
from model import Actor
import tensorflow as tf
import numpy as np

save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a4_REINFORCEMENT/checkpoint/REINFORCEMENT_CarPole'

def calc_return(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1] 
    return np.array(reward_list) # return_list

class Agent(object):
    def __init__(self):
        self.gamma = 0.99
        self.lr = 1e-3
        self.actor = Actor()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def predict(self,observation):
        observation = np.array([observation])
        action_prob = self.actor(observation).numpy()
        action_prob = np.squeeze(action_prob, axis=0)
        return np.random.choice([0,1], p=action_prob)

    def learn(self,observations,actions,reward):
  
        # self.algorithm.learn(
        #     observation_list,
        #     action_list,
        #     calc_return(reward_list,self.gamma)
        # )
        returns = calc_return(reward,self.gamma)
        for step in range(len(returns)):
            observation = np.expand_dims(observations[step],axis=0)
            action = np.expand_dims(actions[step],axis=0)
            Gt = np.expand_dims(returns[step],axis=0)
            with tf.GradientTape() as tape:
                act_prob = self.actor(observation)
                # log_prob = tf.math.log(act_prob)@tf.expand_dims(tf.one_hot(action,2),1)
                log_prob = tf.matmul(tf.math.log(act_prob),tf.transpose(tf.one_hot(action,2)))
                loss_value = -Gt*log_prob
            grads = tape.gradient(
                loss_value,self.actor.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.actor.trainable_weights))

    def save_models(self):
        self.actor.save_weights(save_path + '-actor.ckpt')

    def load_models(self):
        self.actor.load_weights(save_path + '-actor.ckpt')
