from model import Critic,Actor
import numpy as np
import tensorflow as tf
import copy
import pickle

save_path = '/home/ubuntu-1/a7_SAC_v3_Car_910/checkpoint/sac_CarRacing'

class Agent(object):
    def __init__(
        self,
        learning_rate = 3e-4,
        gamma = 0.99,
        tau = 0.003,
        action_dim=1,
        reward_scale=1
    ):
        self.action_dim = action_dim
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor = Actor(self.action_dim)

        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tf.Variable(1.0, dtype=tf.float32)
        self.target_entropy = -tf.constant(action_dim, dtype=tf.float32) # H = -dim(a)

        self.lr = tf.constant(learning_rate,dtype=tf.float32)
        self.gamma = tf.constant(gamma,dtype=tf.float32)
        self.tau = tf.constant(tau,dtype=tf.float32)
        self.reward_scale = tf.constant(reward_scale,dtype=tf.float32)

        self.critic1_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.critic2_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=self.lr)

    def sync_model(self):
        # updating critic network
        target_weights = self.target_critic1.trainable_variables
        weights = self.critic1.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

        target_weights = self.target_critic2.trainable_variables
        weights = self.critic2.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def predict(self,observation):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        action, _ = self.actor.sample_normal(observation)
        return action[0]
        
    def learn(self,transtion_batch):

        observations = tf.convert_to_tensor(transtion_batch[0])
        actions = tf.convert_to_tensor(transtion_batch[1])
        rewards = tf.convert_to_tensor(transtion_batch[2])
        rewards = tf.expand_dims(rewards,1)
        # reward can be normalize
        next_observations = tf.convert_to_tensor(transtion_batch[3])
        dones = tf.convert_to_tensor(transtion_batch[4])
        dones = tf.expand_dims(dones,1)

        # update critic
        next_actions,next_log_pis = self.actor.sample_normal(next_observations)
        q1_next_predict = self.target_critic1(next_observations,next_actions)
        q2_next_predict = self.target_critic2(next_observations,next_actions)
        q_next_predict = tf.math.minimum(q1_next_predict,q2_next_predict)

        soft_q_target = q_next_predict - self.alpha * next_log_pis
        q_target = rewards + (1-dones) * self.gamma * soft_q_target
        with tf.GradientTape() as tape:
            q1_predict = self.critic1(observations,actions)
            loss_critic1 = tf.keras.losses.MSE(q_target,q1_predict)
        grads_critic1 = tape.gradient(
            loss_critic1,self.critic1.trainable_variables
        )
        self.critic1_optimizer.apply_gradients(
            zip(grads_critic1, self.critic1.trainable_variables)
        ) 
        with tf.GradientTape() as tape:
            q2_predict = self.critic2(observations,actions)
            loss_critic2 = tf.keras.losses.MSE(q_target,q2_predict)
        grads_critic2 = tape.gradient(
            loss_critic2,self.critic2.trainable_variables
        )
        self.critic2_optimizer.apply_gradients(
            zip(grads_critic2, self.critic2.trainable_variables)
        )  
        # update policy network
        with tf.GradientTape() as tape:
            new_actions,log_pis = self.actor.sample_normal(observations)
            q1_predict = self.critic1(observations,new_actions)
            q2_predict = self.critic2(observations,new_actions)
            q_predict  = tf.math.minimum(q1_predict,q2_predict)
            loss_actor = self.alpha*log_pis - q_predict
        grads_actor  = tape.gradient(
            loss_actor,self.actor.trainable_variables
        )

        self.actor_optimizer.apply_gradients(
            zip(grads_actor,self.actor.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions, log_pis = self.actor.sample_normal(observations)
            loss_alpha  = -1.0*(
                tf.math.exp(self.log_alpha)*(log_pis + self.target_entropy))
            loss_alpha = tf.nn.compute_average_loss(loss_alpha) 
        variables = [self.log_alpha]
        grads_alpha = tape.gradient(loss_alpha,variables)
        self.alpha_optimizer.apply_gradients(
            zip(grads_alpha, variables)
        )
        self.alpha = tf.math.exp(self.log_alpha)

        self.sync_model()
        return loss_critic1,loss_critic2,loss_actor
    
    def save_models(self):
        
        self.critic1.save_weights(save_path + '-critic1.ckpt')
        self.critic2.save_weights(save_path + '-critic2.ckpt')
        self.actor.save_weights(save_path + '-actor.ckpt')
        data = {'alpha':self.alpha,'log_alpha':self.log_alpha}
        with open(save_path+'-alpha.pickle','wb') as f:
            pickle.dump(data,f)
    
    def load_models(self):

        self.critic1.load_weights(save_path + '-critic1.ckpt').expect_partial()
        self.critic2.load_weights(save_path + '-critic2.ckpt').expect_partial()
        self.actor.load_weights(save_path + '-actor.ckpt').expect_partial()
        with open(save_path+'-alpha.pickle','rb') as f:
            data = pickle.load(f)
            self.alpha = data['alpha']
            self.log_alpha = data['log_alpha']
        

    

       
        
            
            



