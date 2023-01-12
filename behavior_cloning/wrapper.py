import collections
import numpy as np
import gym

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