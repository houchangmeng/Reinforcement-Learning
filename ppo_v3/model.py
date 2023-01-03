import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import tensorflow_probability as tfp


EPSILON = 1e-6 # log(0) protect
LOG_STD_MAX = 1
LOG_STD_MIN = -20

class Critic(Model):
    def __init__(self):
        super().__init__()
        self.f1 = Dense(100,activation='relu')
        self.f2 = Dense(1,activation=None) 

    def call(self,observation):
        x = self.f1(observation)
        v = self.f2(x)
        return v

class Actor(Model):
    def __init__(self,action_dim):
        super().__init__()
        
        self.f1 = Dense(100,activation='relu')
        self.mu_f = Dense(action_dim,activation='tanh')
        self.sigma_f = Dense(action_dim,activation='softplus')

    def call(self,observation):
        x = self.f1(observation)
        mu = self.mu_f(x)
        sigma = self.sigma_f(x)
        sigma = tf.clip_by_value(sigma,1e-4,2.0)
        return mu,sigma
    