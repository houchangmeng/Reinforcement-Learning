from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

class ActorModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(256,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(2,activation='softmax') 

    def call(self,x):
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y

class CriticModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(256,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(2,activation=None)

    def call(self,x):
        x = self.f1(x)
        x = self.f2(x)
        v = self.f3(x)
        return v
