import numpy as np
import collections
import random

class ReplayBuffer(object):
    def __init__(self,max_size=1000) -> None:
        self.buffer = collections.deque(maxlen=max_size)
          
    def sample(self,batch_size):
        
        mini_batch = random.sample(self.buffer, batch_size)
        observation_batch, \
        action_batch, \
        reward_batch, \
        next_observation_batch,\
        done_batch,\
        logprob_batch = [], [], [], [], [],[]

        for experience in mini_batch:
            s, a, r, s_n, done,log_prob = experience
            observation_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_observation_batch.append(s_n)
            done_batch.append(done)
            logprob_batch.append(log_prob)

        return np.array(observation_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_observation_batch).astype('float32'),\
            np.array(done_batch).astype('float32'),\
            np.array(logprob_batch).astype('float32')
       
    def append(self,transtion):
        self.buffer.append(transtion)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()