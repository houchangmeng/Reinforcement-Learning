import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

try:
    env = gym.make("CartPole-v0")
    agent = Agent()
    score_list = []
    average_score_list = []
    total_eposides = 500
    best_score = -1
    for episide in range(total_eposides):
        observation ,_= env.reset()
        episode_step = 0
        score = 0
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, _,_ = env.step(action)
            transition = [observation,action,reward,next_observation,done] # S A R S
            observation = next_observation
            if done:
                break
            score += reward
            agent.learn(transition)
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