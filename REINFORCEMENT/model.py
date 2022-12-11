import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import numpy as np

class Actor(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(64,activation='relu')
        self.f2 = Dense(64,activation='relu')
        self.f3 = Dense(2,activation='softmax') 

    def call(self,x):
        x = self.f1(x)
        x = self.f2(x)
        action_prob = self.f3(x)
        return action_prob
