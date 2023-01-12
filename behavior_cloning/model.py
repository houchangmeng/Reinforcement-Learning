import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import tensorflow_probability as tfp
import gym
import collections

EPSILON = 1e-6 # log protect
LOG_STD_MAX = 1
LOG_STD_MIN = -20
MAX_ACTION = 2

class CarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.reward_deque = collections.deque(maxlen=200) #
        self.low = self.action_space.low
        self.high = self.action_space.high
        self._reverse_low  =  np.array([-1,-1,-1])
        self._reverse_high =  np.array([ 1, 1, 1])
        self.action_repeat = 4
        self.frame = None

    def _action(self, action):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
        action = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        action = np.clip(action, self.low, self.high)
        
        return action

    def _reverse_action(self, action):
        """Map action space [low_bound, high_bound] of environment to 
        new action space (ex:model output)[-1, 1] 
        """
        action = -1 + 2 * (action - self.low) / (self.high - self.low) 
        action = np.clip(action, self._reverse_low, self._reverse_high)

        return action

    def step(self, action):
        action = self._action(action)
        total_reward = 0
        done = False
        truncated = False
        for _ in range(self.action_repeat):
            next_observation, reward, done, truncated, info  = self.env.step(action)
            # 1. 处理奖励
            if truncated:
                reward += 100
                pass
            # green penalty
            mean_green = np.mean(next_observation[:,:,1])
            if mean_green >= 185.0:
                reward -= 0.05
            mean_reward = np.mean(self.reward_deque)
            if mean_reward < (-0.0999):
                # reward -= 1
                # done = True
                pass
            total_reward += reward
            self.reward_deque.append(reward)
            # 2. done process
            if done or truncated:
                break
        next_observation_gray = self.process_image(next_observation)
        self.frame = np.concatenate([self.frame,next_observation_gray],axis=-1)
        self.frame = np.delete(self.frame,obj=0,axis=-1)
        return self.frame, total_reward, done, truncated, info
    
    def reset(self):
        self.reward_deque.clear()
        observation,info = self.env.reset()
        obs_ = self.process_image(observation)
        self.frame = np.concatenate([obs_,obs_,obs_,obs_],axis=-1)
        
        return self.frame,info
    
    def sample_action(self):
        action = self.env.action_space.sample()
        action = self._reverse_action(action)
        return action

    @staticmethod
    def process_image(observation):
        a = np.array([0.299,0.587,0.114])
        new_observation = np.dot(observation[:84,:84,:],a) # 裁剪 灰度
        new_observation = new_observation/128.0 - 1.0 # 正则化
        new_observation = np.expand_dims(new_observation,axis=2) # 84*84*1
        return new_observation

class ConvNN(Model):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(8, kernel_size=4, strides=2, padding='same'),# 卷积层
            Activation('relu'),# 激活层
            Conv2D(16, kernel_size=3, strides=2, padding='same'),# 卷积层
            Activation('relu'),# 激活层
            Conv2D(32, kernel_size=3, strides=2, padding='same'),# 卷积层
            Activation('relu'),# 激活层
            Conv2D(64, kernel_size=3, strides=2, padding='same'),# 卷积层
            Activation('relu'),# 激活层
            Conv2D(128, kernel_size=3, strides=1, padding='same'),# 卷积层
            Activation('relu'),# 激活层
            Conv2D(256, kernel_size=3, strides=1, padding='same'),# 卷积层
            Activation('relu'),# 激活层
        ])

    def call(self, x):
        x = self.model(x, training=False) 
        return x
        
class Critic(Model):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = ConvNN()
        self.flatten = Flatten()

        self.o1 = Dense(128,activation='relu')
        self.a1 = Dense(128,activation='relu')

        self.f1 = Dense(128,activation='relu') 
        self.f2 = Dense(128,activation='relu') 
        self.f3 = Dense(1,activation=None) 

    def call(self,observation,action):
        o = self.c1(observation)
        o = self.flatten(o)

        o = self.o1(o)
        a = self.a1(action)

        x = self.f1(tf.concat([o, a], axis=1))
        x = self.f2(x)
        q = self.f3(x)
        return q

class Actor(Model):
    def __init__(self,action_dim) -> None:
        super().__init__()
        self.c1 = ConvNN()
        
        self.flatten = Flatten()
        
        self.f1 = Dense(128,activation='relu')
        self.f2 = Dense(128,activation='relu')
        self.mu_f = Dense(action_dim,activation= None)
        self.log_sigma_f = Dense(action_dim,activation=None)

    def call(self,observation):
        o = self.c1(observation)

        o = self.flatten(o)

        x = self.f1(o)
        x = self.f2(x)
        mu = self.mu_f(x)
        log_sigma = self.log_sigma_f(x)
        log_sigma = tf.clip_by_value(log_sigma,LOG_STD_MIN,LOG_STD_MAX)
        sigma = tf.math.exp(log_sigma)
        return mu,sigma
    
    def sample_normal(self,observation):
        mu,sigma = self.call(observation)
        
        # 没有重参数
        # normal = tfp.distributions.Normal(mu, sigma) 
        # u      = normal.sample()
        # log_pi = normal.log_prob(u)
        # action = tf.tanh(u)
         
        # 重参数
        normal_ = tfp.distributions.Normal(tf.zeros(mu.shape),tf.ones(sigma.shape))
        e = normal_.sample()
        log_pi = normal_.log_prob(e)
        action = tf.tanh(mu + e * sigma)
        
        log_pi = log_pi - tf.math.log((1-action**2 + EPSILON))
        log_pi = tf.reduce_sum(log_pi,axis=1,keepdims=True)
        return action,log_pi
