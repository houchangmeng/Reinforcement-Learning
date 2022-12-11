
import numpy as np
import tensorflow as tf
from model import CriticModel,ActorModel
import copy

class Agent(object):
    '''
    TD3
    1. Clipped Double-Q Learning
    2. “Delayed” Policy Updates.
    3. Target Policy Smoothing.
    '''
    def __init__(
        self,action_dim = 1,
        e_greedy = 0.1,
        e_greedy_decay =0.001
        ):
        self.action_dim = action_dim
        self.min_action = -1
        self.max_action = 1 

        self.e_greedy = e_greedy
        self.e_greedy_decay = e_greedy_decay
        self.learn_num = 0

        self.critic_model1 = CriticModel()
        self.critic_model2 = CriticModel()
        self.target_critic_model1 = copy.deepcopy(self.critic_model1)
        self.target_critic_model2 = copy.deepcopy(self.critic_model2)

        self.actor_model = ActorModel()
        self.target_actor_model = copy.deepcopy(self.actor_model)

        self.lr = 0.0003
        self.gamma = 0.9
        self.tau = 0.005
        self.total_it = 0
        self.policy_freq = 2

        self.critic_model_optimizer1= tf.optimizers.Adam(learning_rate=self.lr)
        self.critic_model_optimizer2= tf.optimizers.Adam(learning_rate=self.lr)
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

        with tf.GradientTape(persistent=True) as tape:
            next_actions = self.target_actor_model(next_observations)
            q_next1 = self.target_critic_model1(next_observations,next_actions)
            q_next2 = self.target_critic_model2(next_observations,next_actions)
            q_target = rewards + (1-dones)*self.gamma * tf.minimum(q_next1,q_next2)
            
            q_predict1 = self.critic_model1(observations,actions)
            loss_critic1 = tf.keras.losses.MSE(q_predict1, q_target)
            q_predict2 = self.critic_model2(observations,actions)
            loss_critic2 = tf.keras.losses.MSE(q_predict2, q_target)
        
        grads_critic1 = tape.gradient(
            loss_critic1,self.critic_model1.trainable_variables
        )
        self.critic_model_optimizer1.apply_gradients(
            zip(grads_critic1, self.critic_model1.trainable_variables)
        )
        grads_critic2 = tape.gradient(
            loss_critic2,self.critic_model2.trainable_variables
        )
        self.critic_model_optimizer2.apply_gradients(
            zip(grads_critic2, self.critic_model2.trainable_variables)
        )

        self.total_it += 1 
        if self.total_it % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                actions = self.actor_model(observations)
                q = self.critic_model1(observations,actions)
                loss_actor = -tf.reduce_mean(q)
                
            grads_actor  = tape.gradient(
                loss_actor,self.actor_model.trainable_variables
            )

            self.actor_model_optimizer.apply_gradients(
                zip(grads_actor, self.actor_model.trainable_variables)
            )

        self.sync_model()

        self.learn_num+=1
        new_greedy = 0.1 + 0.88 * (1 - np.exp(-self.learn_num/100))
        if self.e_greedy < new_greedy:
            self.e_greedy = new_greedy

    def sync_model(self):
        # updating critic network 1
        target_weights1 = self.target_critic_model1.trainable_variables
        weights1 = self.critic_model1.trainable_variables
        for (a, b) in zip(target_weights1, weights1):
            a.assign(b * self.tau + a * (1 - self.tau))
        
        # updating critic network 2
        target_weights2 = self.target_critic_model2.trainable_variables
        weights2 = self.critic_model2.trainable_variables
        for (a, b) in zip(target_weights2, weights2):
            a.assign(b * self.tau + a * (1 - self.tau))
        
        # updating actor network
        target_weights = self.target_actor_model.trainable_variables
        weights = self.actor_model.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
    
    def save_models(self):
        checkpoint_save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a9_TD3/checkpoint/td3_Pendulum.ckpt'
        self.actor_model.save_weights(checkpoint_save_path)

    def load_models(self):
        checkpoint_save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a9_TD3/checkpoint/td3_Pendulum.ckpt'
        self.actor_model.load_weights(checkpoint_save_path)
            
            



