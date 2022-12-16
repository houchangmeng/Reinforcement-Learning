import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

class QModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(256,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.fv = Dense(1,activation=None) 
        self.fa = Dense(2,activation=None) 
        

    def call(self,observation):
        x = self.f1(observation)
        x = self.f2(x)
        v = self.fv(x)
        a = self.fa(x)
        a_mean = tf.expand_dims(tf.reduce_mean(a,axis=1),axis=1)
        q = v + a - a_mean
        
        return q
