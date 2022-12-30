import gym
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from gym.wrappers import RescaleAction
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

try:
    env = RescaleAction(gym.make("Pendulum-v1",render_mode='human'),min_action=-1,max_action=1)
    action_dim=env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(action_dim=action_dim)
    # load model
    agent.load_models()
    score_list = []
    average_score_list = []
    total_eposide = 10000
    for episide in range(total_eposide):
        steps = 0
        score = 0
        observation,_ = env.reset()
        while True:
            action ,_= agent.predict(observation)
            next_observation, reward, done, _ , _ = env.step(action)
            observation = next_observation
            score += reward
            steps +=1
            env.render()
            if done or (steps >= 200):
                break
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        print(  'episode %d/%d'% (episide,total_eposide),
                'score %.1f' % score, 
                'avg_score %.1f' % average_score)
finally:
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()
    env.close()
