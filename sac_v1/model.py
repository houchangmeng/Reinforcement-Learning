import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import \
    Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Concatenate
import tensorflow_probability as tfp

EPSILON = 1e-6
LOG_STD_MAX = 1
LOG_STD_MIN = -20

class Critic(Model):
    def __init__(self,action_dim) -> None:
        super().__init__()
        self.f1 = Dense(512,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(128,activation='relu')
        self.f4 = Dense(action_dim,activation='linear') 

    def call(self,observation,action):
        x = self.f1(tf.concat([observation, action], axis=1))
        x = self.f2(x)
        x = self.f3(x)
        q = self.f4(x)
        return q

class Value(Model):
    def __init__(self,action_dim) -> None:
        super().__init__()
        self.f1 = Dense(512,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(128,activation='relu')
        self.f4 = Dense(action_dim,activation='linear') 

    def call(self,observation):
        x = self.f1(observation)
        x = self.f2(x)
        x = self.f3(x)
        v = self.f4(x)
        return v

class Actor(Model):
    def __init__(self,action_dim) -> None:
        super().__init__()
        
        self.f1 = Dense(512,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(128,activation='relu')
        self.mu_f = Dense(action_dim,activation=None)
        self.log_sigma_f = Dense(action_dim,activation=None) 

    def call(self,observation):
        x = self.f1(observation)
        x = self.f2(x)
        x = self.f3(x)
        mu = self.mu_f(x)
        log_sigma = self.log_sigma_f(x)
        log_sigma = tf.clip_by_value(log_sigma,LOG_STD_MIN,LOG_STD_MAX)
        sigma = tf.math.exp(log_sigma)
        return mu,sigma

    def sample_normal(self, observation):
        mu,sigma = self.call(observation)
        normal   = tfp.distributions.Normal(mu, sigma) 
        u        = normal.sample()
 
        action   = tf.math.tanh(u) 
        log_pis  = normal.log_prob(u)
        log_pis -= tf.math.log((1.0 - tf.math.pow(action, 2) + 1e-6))
        log_pis  = tf.math.reduce_sum(log_pis,axis=1,keepdims=True)
        return action,log_pis
       
        

        
