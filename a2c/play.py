import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    env = gym.make("CartPole-v0",render_mode="human")
    agent = Agent()
    agent.load_models()
    score_list = []
    average_score_list = []
    total_eposides = 500

    for episide in range(total_eposides):
        observation,_ = env.reset()
        episode_step = 0
        score = 0
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, _,_ = env.step(action)
            observation = next_observation
            if done:
                break
            score += reward
            env.render()
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)

        print(  'episode %d/%d'% (episide,total_eposides),
                'score %.1f' % score, 
                'avg_score %.1f' % average_score)

finally:
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()
    env.close()