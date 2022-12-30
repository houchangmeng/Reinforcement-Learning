from model import Critic,Actor
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import copy
tfd = tfp.distributions
demo_path = "/home/ubuntu-1/Learning/ReinforcementLearning/demo/a8_PPO_v5"

class Agent(object):
    def __init__(
        self,
        learning_rate = 1e-4,
        gamma = 0.9,
        action_dim=1,
    ):
        self.action_dim = action_dim
        self.critic = Critic()
        self.actor = Actor(self.action_dim)
     
        self.lr = tf.constant(learning_rate,dtype=tf.float32)
        self.gamma = tf.constant(gamma,dtype=tf.float32)
        self.clip_param = 0.2

        self.critic_optimizer = tf.optimizers.Adam(
            learning_rate=2e-4,clipvalue=1.0)
        self.actor_optimizer = tf.optimizers.Adam(
            learning_rate=1e-4,clipvalue=1.0)

    def predict(self,observation):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu, sigma = self.actor(observation)
        pi = tfd.Normal(mu, sigma) 
        action = pi.sample()
        action = tf.clip_by_value(action,-1.0,1.0)
        log_pi = pi.log_prob(action)
        
        return action[0],log_pi[0]
        
    def learn(self,transtion_batch):

        observations = tf.convert_to_tensor(transtion_batch[0])
        actions = tf.convert_to_tensor(transtion_batch[1])
        rewards = tf.convert_to_tensor(transtion_batch[2])
        rewards = tf.expand_dims(rewards,axis=-1)  

        next_observations = tf.convert_to_tensor(transtion_batch[3])
        dones = tf.convert_to_tensor(transtion_batch[4])
        dones = tf.expand_dims(dones,axis=-1) 

        # old_mu,old_sigma = self.actor(observations)
        # old_pi = tfd.Normal(loc=old_mu,scale = old_sigma)
        # old_log_pis = old_pi.log_prob(actions)

        old_log_pis = tf.convert_to_tensor(transtion_batch[5])

        targets = rewards + self.gamma * self.critic(next_observations) *(1.0 -dones ) 
        advantages = targets - self.critic(observations)

        for _ in range(10):
            with tf.GradientTape() as tape:
                # _,log_pis = self.actor.sample_normal(observations)
                mu,sigma = self.actor(observations)
                pi = tfd.Normal(mu, sigma) 
                log_pis = pi.log_prob(actions)

                ratio = tf.exp(log_pis - old_log_pis)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(
                    ratio,1.0 - self.clip_param,1.0 + self.clip_param) * advantages
                loss_actor = - tf.reduce_mean(tf.math.minimum(surr1,surr2))
            grads_actor = tape.gradient(
                loss_actor,self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(grads_actor, self.actor.trainable_variables)) 

            with tf.GradientTape() as tape:
                v_value = self.critic(observations)
                loss_critic = tf.keras.losses.MSE(targets,v_value)
            grads_critic = tape.gradient(
                loss_critic,self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(grads_critic, self.critic.trainable_variables))

    def save_models(self):
        
        self.actor.save_weights(demo_path+"/checkpoint/ppo_actor.ckpt")
        self.critic.save_weights(demo_path+"/checkpoint/ppo_critic.ckpt")
    
    def load_models(self):
        self.actor.load_weights(demo_path+"/checkpoint/ppo_actor.ckpt")
        self.critic.load_weights(demo_path+"/checkpoint/ppo_critic.ckpt")
    

       
        
            
            



