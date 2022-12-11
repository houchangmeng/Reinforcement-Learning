import gym
from gym.wrappers import RescaleAction
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from replay_buffer import ReplayBuffer


continue_train = True
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    env = RescaleAction(gym.make("Pendulum-v1"),min_action=-1,max_action=1)
    agent = Agent(action_dim=1)
    if continue_train:
        agent.load_models()
    replay_buffer = ReplayBuffer(100000)
    batch_size = 256
    score_list = []
    average_score_list = []
    best_score = -2000
    total_eposide = 10000
    learn_num = 20
    for episide in range(total_eposide):
        observation,_ = env.reset()
        steps = 0
        score = 0
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, _ , _ = env.step(action)
            transtion = [observation,action,reward,next_observation,done]
            replay_buffer.append(transtion)
            observation = next_observation
            score += reward
            steps+=1
            #env.render()
            if done or (steps >=200):
                break
        if len(replay_buffer)> batch_size:
            for _ in range(learn_num):
                agent.learn(replay_buffer.sample(batch_size))
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