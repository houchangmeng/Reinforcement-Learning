import gym
import numpy as np
from gym.wrappers import RescaleAction
import matplotlib.pyplot as plt
from agent import Agent

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try: 
    env = RescaleAction(gym.make("Pendulum-v1",render_mode='human'),min_action=-1,max_action=1)
    agent = Agent(action_dim=env.action_space.shape[0])
    # load model
    agent.load_models()
    score_list = []
    average_score_list = []
    total_eposide = 10000
    for episide in range(total_eposide):
        observation,_ = env.reset()
        steps = 0
        score = 0
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, _ , _ = env.step(action)
            observation = next_observation
            score += reward
            steps +=1
            env.render()
            if done or steps>200:
                break
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        print(  'episode %d/%d'% (episide,total_eposide),
                'score %.1f' % score, 
                'avg_score %.1f' % average_score)    
finally:
    env.close()
    plt.figure()
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()

