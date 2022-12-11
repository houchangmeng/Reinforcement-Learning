from model import Critic,Actor,Value
import tensorflow as tf
import numpy as np
import copy
import tensorflow.keras as keras

save_path = "/home/ubuntu-1/Learning/ReinforcementLearning/demo/a7_SAC_v1/"

class Agent(object):
    def __init__(
        self,
        learning_rate = 3e-4,
        gamma = 0.99,
        tau = 0.005,
        action_dim=1,
        reward_scale=10,
        alpha = 0.2
    ):
        self.action_dim = action_dim
        self.actor = Actor(action_dim)
        self.critic1 = Critic(action_dim)
        self.critic2 = copy.deepcopy(self.critic1)
        self.value = Value(action_dim)
        self.target_value = copy.deepcopy(self.value)
        
        self.lr = tf.constant(learning_rate,dtype=tf.float32)
        self.gamma = tf.constant(gamma,dtype=tf.float32)
        self.tau = tf.constant(tau,dtype=tf.float32)
        self.alpha = tf.constant(alpha,dtype=tf.float32)
        self.reward_scale = tf.constant(reward_scale,dtype=tf.float32)

        self.actor_optimizer   = tf.optimizers.Adam(learning_rate=self.lr)
        self.critic1_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.critic2_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.value_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        

    def predict(self,observation):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        action, _ = self.actor.sample_normal(observation)
        return action[0]
        
    def learn(self,transtion_batch):
        
        observations = tf.convert_to_tensor(transtion_batch[0])
        actions = tf.convert_to_tensor(transtion_batch[1])
        rewards = tf.convert_to_tensor(transtion_batch[2])
        rewards = tf.expand_dims(rewards,axis=1)
        next_observations = tf.convert_to_tensor(transtion_batch[3])
        dones = tf.convert_to_tensor(transtion_batch[4])
        dones = tf.expand_dims(dones,axis=1)
        # update v network
        with tf.GradientTape() as tape_v:
            v_predict = self.value(observations)
            
            new_actions,log_pis = self.actor.sample_normal(observations)
            q1_new = self.critic1(observations,new_actions)
            q2_new = self.critic2(observations,new_actions)
            q_predict = tf.math.minimum(q1_new,q2_new)

            v_target  = q_predict-self.alpha*log_pis
            loss_value= keras.losses.MSE(v_target,v_predict)
        grads_value = tape_v.gradient(
            loss_value,self.value.trainable_variables
        )
        self.value_optimizer.apply_gradients(
            zip(grads_value, self.value.trainable_variables)
        )
        # update policy network
        with tf.GradientTape() as tape:
            new_actions,log_pis = self.actor.sample_normal(observations)
            q1_new = self.critic1(observations,new_actions)
            q2_new = self.critic2(observations,new_actions)
            q_predict = tf.math.minimum(q1_new,q2_new)
            loss_actor = self.alpha*log_pis - q_predict
            loss_actor = tf.math.reduce_mean(loss_actor) 
        grads_actor  = tape.gradient(
            loss_actor,self.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(grads_actor, self.actor.trainable_variables)
        )
       
        # update q network
        with tf.GradientTape(persistent=True) as tape:
            v_target_ = self.target_value(next_observations)
            q_target = self.reward_scale*rewards + (1-dones) * self.gamma*v_target_
            q1 = self.critic1(observations,actions)
            q2 = self.critic2(observations,actions)
            loss_critic1 = keras.losses.MSE(q_target, q1)
            loss_critic2 = keras.losses.MSE(q_target, q2)
        grads_critic1 = tape.gradient(
            loss_critic1,self.critic1.trainable_variables
        )
        self.critic1_optimizer.apply_gradients(
            zip(grads_critic1, self.critic1.trainable_variables)
        )
        grads_critic2 = tape.gradient(
            loss_critic2,self.critic2.trainable_variables
        )
        self.critic2_optimizer.apply_gradients(
            zip(grads_critic2, self.critic2.trainable_variables)
        ) 

        self.sync_model()
        
    def sync_model(self):
        # updating target v network
        target_weights = self.target_value.trainable_variables
        weights = self.value.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
    
    def save_models(self):
        checkpoint_save_path = save_path+"/checkpoint/sac_v1.ckpt"
        self.actor.save_weights(checkpoint_save_path)
    
    def load_models(self):
        checkpoint_save_path = save_path+"/checkpoint/sac_v1.ckpt"
        self.actor.load_weights(checkpoint_save_path)


        

            



