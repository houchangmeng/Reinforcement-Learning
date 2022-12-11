
import numpy as np
import tensorflow as tf
from model import CriticModel,ActorModel
import copy

class Agent(object):
    def __init__(
        self,
        action_dim = 1,
        e_greedy = 0.1,
        e_greedy_decay =0.001
        ):
        self.action_dim = action_dim
        self.min_action = -1
        self.max_action = 1

        self.e_greedy = e_greedy
        self.e_greedy_decay = e_greedy_decay
        self.learn_num = 0

        self.critic_model = CriticModel()
        self.target_critic_model = copy.deepcopy(self.critic_model)

        self.actor_model = ActorModel()
        self.target_actor_model = copy.deepcopy(self.actor_model)

        self.lr = 0.0001
        self.gamma = 0.9
        self.tau = 0.005

        self.critic_model_optimizer= tf.optimizers.Adam(learning_rate=self.lr)
        self.actor_model_optimizer= tf.optimizers.Adam(learning_rate=self.lr)

        
    def predict(self,observation):
        
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        action = self.actor_model(observation)
        
        if np.random.random()>self.e_greedy:
            noise = np.random.normal(0,0.05)
            action = np.clip(action+noise,self.min_action,self.max_action)
        else:
            action = np.clip(action,self.min_action,self.max_action)
        return np.squeeze(action,axis=1)
        
    def learn(self,transtion_batch):

        observations = tf.convert_to_tensor(transtion_batch[0])
        actions = tf.convert_to_tensor(transtion_batch[1])
        rewards = tf.convert_to_tensor(transtion_batch[2])
        rewards = tf.expand_dims(rewards,1)
        next_observations = tf.convert_to_tensor(transtion_batch[3])
        dones = tf.convert_to_tensor(transtion_batch[4])
        dones = tf.expand_dims(dones,1)
        # update critic
        with tf.GradientTape() as tape:
            next_actions = self.target_actor_model(next_observations)
            q_next = self.target_critic_model(next_observations,next_actions)
            q_target = rewards + (1-dones)*self.gamma * q_next
            
            q_predict = self.critic_model(observations,actions)
            loss_critic = tf.keras.losses.MSE(q_predict, q_target)
        grads_critic = tape.gradient(
            loss_critic,self.critic_model.trainable_variables)
        self.critic_model_optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))
        
        # update policy
        with tf.GradientTape() as tape:
            actions = self.actor_model(observations)
            q = self.critic_model(observations,actions)
            loss_actor = -tf.reduce_mean(q)
        grads_actor  = tape.gradient(
            loss_actor,self.actor_model.trainable_variables)
        self.actor_model_optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables) )

        self.sync_model()

        # reduce exploration
        self.learn_num+=1
        new_greedy = 0.1 + 0.88 * (1 - np.exp(-self.learn_num/100))
        if self.e_greedy < new_greedy:
            self.e_greedy = new_greedy

    def sync_model(self):
        # updating critic network
        target_weights = self.target_critic_model.trainable_variables
        weights = self.critic_model.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
        
        # updating Actor network
        target_weights = self.target_actor_model.trainable_variables
        weights = self.actor_model.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
    
    def save_models(self):
        checkpoint_save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a6_DDPG/checkpoint/ddpg_Pendulum.ckpt'
        self.actor_model.save_weights(checkpoint_save_path)

    def load_models(self):
        checkpoint_save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a6_DDPG/checkpoint/ddpg_Pendulum.ckpt'
        self.actor_model.load_weights(checkpoint_save_path)
            
            



