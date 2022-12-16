import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

class QModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(256,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(2,activation=None) 

    def call(self,observation):
        x = self.f1(observation)
        x = self.f2(x)
        q = self.f3(x)
        return q
