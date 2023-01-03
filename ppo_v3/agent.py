
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from model import Actor,Critic

import numpy as np
import tensorflow as tf
import copy
import tensorflow_probability as tfp
tfd = tfp.distributions

demo_path = "/home/ubuntu-1/Learning/ReinforcementLearning/demo/a8_PPO_v3"

class Agent(object):
    '''
    Basic PPO Agent : gae.
    '''
    def __init__(
        self,
        observation_shape=(3,),
        action_dim=1,
        learning_rate = 1e-4,
        gamma = 0.9,
    ):
        self.action_dim = action_dim
        self.critic = Critic()
        self.actor = Actor(action_dim=action_dim)
        
        self.lr = tf.constant(learning_rate,dtype=tf.float32)
        self.gamma = tf.constant(gamma,dtype=tf.float32)
        self.clip_param = 0.2
        self.lamda = 0.95

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=2e-4,clipvalue=1.0)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=1e-4,clipvalue=1.0)

    def predict(self,observation):
        observation = tf.convert_to_tensor([observation],dtype=tf.float32)
        mu,sigma = self.actor(observation)
        pi = tfd.Normal(loc=mu, scale=sigma) 
        action = pi.sample()

        action = np.clip(action[0],-1.0,1.0)
        log_pi = pi.log_prob(action)
        return action,log_pi.numpy()[0]

    def learn(self,observation_list,action_list,reward_list,next_observation_list,done_list,log_prob_list):
        
        observations = tf.convert_to_tensor(observation_list)
        actions = tf.convert_to_tensor(action_list)
        rewards = tf.expand_dims(tf.convert_to_tensor(reward_list,dtype=tf.float32),axis=-1)
        next_observations = tf.convert_to_tensor(next_observation_list)
        dones = tf.expand_dims(tf.convert_to_tensor(done_list,dtype=tf.float32),axis=-1)
        old_pi_log_prob = tf.convert_to_tensor(log_prob_list)

        value_next_obs = self.critic(next_observations)
        value_obs = self.critic(observations)

        adv = []
        gae = 0
        for r, d, vs_, vs in zip(rewards[::-1], dones[::-1],value_next_obs[::-1],value_obs[::-1]):
            delta = r + self.gamma * (1.0 - d) * vs_ - vs
            gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
            adv.insert(0, gae)
        advantages = tf.convert_to_tensor(adv)
        targets = adv + value_obs

        for _ in range(10):
            with tf.GradientTape() as tape:
                mu,sigma = self.actor(observations)
                pi = tfd.Normal(mu,sigma)
                pi_log_prob = pi.log_prob(actions)
                ratio = tf.exp(pi_log_prob - old_pi_log_prob)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)*advantages
                actor_loss = - tf.reduce_mean(tf.math.minimum(surr1,surr2))
            grads_actor  = tape.gradient(
                actor_loss,self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(grads_actor,self.actor.trainable_variables))

            with tf.GradientTape() as tape:
                v_value = self.critic(observations)
                critic_loss = tf.keras.losses.MSE(targets,v_value)
            grads_critic = tape.gradient(
                critic_loss,self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(grads_critic, self.critic.trainable_variables))
 
    def save_models(self):
        
        self.actor.save_weights(demo_path+"/checkpoint/ppo_actor.ckpt")
        self.critic.save_weights(demo_path+"/checkpoint/ppo_critic.ckpt")
    
    def load_models(self):

        self.actor.load_weights(demo_path+"/checkpoint/ppo_actor.ckpt")
        self.critic.load_weights(demo_path+"/checkpoint/ppo_critic.ckpt")
