import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import \
    Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Concatenate

class CriticModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(256,activation='relu')
        self.f2 = Dense(512,activation='relu') 
        self.f3 = Dense(512,activation='relu') 
        self.f4 = Dense(1,activation=None) 

    def call(self,observation,action):
        x = self.f1(tf.concat([observation, action], axis=1))
        x = self.f2(x)
        x = self.f3(x)
        q = self.f4(x)
        return q

class ActorModel(Model):
    def __init__(self) -> None:
        super().__init__()
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.f1 = Dense(512,activation='relu')
        self.f2 = Dense(512,activation='relu')
        self.f3 = Dense(1,activation='tanh',kernel_initializer=last_init) # 输出动作

    def call(self,observation):
        x = self.f1(observation)
        x = self.f2(x)
        a = self.f3(x)
        return a
