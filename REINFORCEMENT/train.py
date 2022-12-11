import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

try:
    env = gym.make("CartPole-v0")
    agent = Agent()
    batch_size = 64
    best_score = 100
    score_list = []
    average_score_list = []
    observation_list = []
    action_list = []
    reward_list = []
    total_eposides = 500
    for episide in range(total_eposides):
        observation,_ = env.reset()
        episode_step = 0
        score = 0
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, _,_ = env.step(action)
            score += reward
            #env.render()
            observation_list.append(observation)
            action_list.append(action)
            reward_list.append(reward)
            observation = next_observation
            if done:
                break
        agent.learn(observation_list,action_list,reward_list)
        observation_list.clear()
        action_list.clear()
        reward_list.clear()
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        print(  'episode %d/%d'% (episide,total_eposides),
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