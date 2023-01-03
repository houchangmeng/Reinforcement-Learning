import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
from gym.wrappers import RescaleAction
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
continue_train = False
try:
    env = RescaleAction(gym.make("Pendulum-v1").unwrapped,max_action=1,min_action=-1)
    action_dim=env.action_space.shape[0]
    observation_shape = env.observation_space.shape
    agent = Agent(observation_shape=observation_shape,action_dim=action_dim)
    if continue_train:
        agent.load_models()

    batch_size = 32
    best_score = -400
    score_list = []
    average_score_list = []
    total_eposide = 3000
    episode_length = 256
    for episide in range(total_eposide):
        observation_list = []
        action_list = []
        reward_list = []
        next_observation_list = []
        done_list = []
        log_prob_list = []
        score = 0
        observation,_ = env.reset()
        for steps in range(episode_length):
            action,log_prob  = agent.predict(observation)
            next_observation, reward, done, _ , _ = env.step(action)
            
            observation_list.append(observation)
            action_list.append(action)
            reward_list.append((reward + 8.0)/8.0)
            next_observation_list.append(next_observation)
            done_list.append(float(done))
            log_prob_list.append(log_prob)

            observation = next_observation
            score += reward

            if (steps + 1) % batch_size == 0 or steps == episode_length-1:
                agent.learn(observation_list,action_list,reward_list,next_observation_list,done_list,log_prob_list)
                observation_list = []
                action_list = []
                reward_list = []
                next_observation_list = []
                done_list = []
                log_prob_list = []

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
