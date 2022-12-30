import gym
from gym.wrappers import RescaleAction
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from replay_buffer import ReplayBuffer

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    env = RescaleAction(gym.make("Pendulum-v1").unwrapped,min_action=-1,max_action=1)
    action_dim=env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(action_dim=action_dim)
    replay_buffer = ReplayBuffer(max_size=1000)
    batch_size = 32
    best_score = -500
    score_list = []
    average_score_list = []
    total_eposide = 3000
    episode_length = 256
    for episide in range(total_eposide):
        steps = 0
        score = 0
        observation,_ = env.reset()
        while True:
            action,log_prob = agent.predict(observation)
            next_observation, reward, done, _ , _ = env.step(action)
            transtion = [observation,action,(reward+8.0)/8.0, next_observation, done,log_prob]
            
            replay_buffer.append(transtion)
            observation = next_observation
            score += reward
            steps+=1
            if done or (steps >= episode_length):
                break
        if len(replay_buffer) >= batch_size:
            for _ in range(5):
                agent.learn(replay_buffer.sample(batch_size=32))
            
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        print(  'episode %d/%d'% (episide,total_eposide),
                'score %.1f' % score, 
                'avg_score %.1f' % average_score)
        if average_score > best_score:
            best_score = average_score
            agent.save_models()
        
finally:
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()
    env.close()
