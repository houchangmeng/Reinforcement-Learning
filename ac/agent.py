import tensorflow as tf
import numpy as np
from tensorflow import keras
from model import ActorModel,CriticModel

save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a5_ActorCritic/checkpoint/ac_CarPole'

class Agent(object):
    def __init__(self) -> None:
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.actor_model  = ActorModel()
        self.critic_model = CriticModel()
        self.actor_optimizer  = keras.optimizers.Adam(learning_rate=self.learning_rate )
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate )

    def predict(self,observation):
        observation = np.array([observation])
        action_prob = self.actor_model(observation).numpy()
        action_prob = np.squeeze(action_prob)
        return np.random.choice([0,1],p=action_prob)

    def learn(self,transition):
        
        observation = np.expand_dims(transition[0],axis=0)
        action =  np.expand_dims(transition[1],axis=0)
        reward = np.expand_dims(transition[2],axis=0)
        next_observation = np.expand_dims(transition[3],axis=0)
        done = np.expand_dims(float(transition[4]),axis=0)

        with tf.GradientTape() as tape1,\
             tf.GradientTape() as tape2:
            q_value = self.critic_model(observation,training=True)
            q_next_value = self.critic_model(next_observation,training=True)
            delta = reward + self.gamma*q_next_value*(1-int(done)) - q_value
            critic_loss = delta**2

            act_prob = self.actor_model(observation,training=True)
            log_prob = tf.matmul(tf.math.log(act_prob),tf.transpose(tf.one_hot(action,2)))
            # update policy gradient with baseline q target is baseline
            # this isn't A2C
            delta_ = tf.matmul(delta,tf.transpose(tf.one_hot(action,2)))
            actor_loss = -delta_*log_prob
            
        grads_actor = tape1.gradient(
            actor_loss,self.actor_model.trainable_variables)
        grads_critic = tape2.gradient(
            critic_loss,self.critic_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads_actor, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(grads_critic, self.critic_model.trainable_variables))
    
    def save_models(self):
        self.actor_model.save_weights(save_path + '-actor.ckpt')

    def load_models(self):
        self.actor_model.load_weights(save_path + '-actor.ckpt')
        
