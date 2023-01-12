import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from agent import Agent
from replay_buffer import ReplayBuffer
from wrapper import CarWrapper
import os


# https://notanymike.github.io/Solving-CarRacing/

continue_train = False
pre_exploration = True
use_cpu = False

if use_cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def exploration(env,replay_buffer):
    episode = 0
    while True:
        episode += 1
        observation,_ = env.reset()
        steps = 0
        while True:
            action = env.sample_action()
            next_observation, reward, done, truncated, _ = env.step(action)
            state = [observation,action,reward,next_observation,done]
            replay_buffer.append(state)
            observation = next_observation
            steps += 1
            if done or truncated:
                break
            if len(replay_buffer) > 256:
                return
        print(f'explorating: episode {episode}, buffer length {len(replay_buffer)} / {replay_buffer.length}') 

try:
    start_time = time.time()
    batch_size = 256
    score_list = []
    average_score_list = []
    total_eposide = 10000
    best_score = 700
    learn_number = 20
    best_alpha = None
    f = open('log.txt','w')

    env = CarWrapper(gym.make('CarRacing-v2'))#,render_mode='human'))# ))#
    action_dim=env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(action_dim=action_dim)
    if continue_train:
        print('\n -----Continue Train----- \n')
        agent.load_models()
    replay_buffer = ReplayBuffer(40000)
    if pre_exploration:
        print('\n -----Exploration   ----- \n')
        exploration(env,replay_buffer)

    for episide in range(total_eposide):
        steps = 0
        score = 0
        observation,_ = env.reset()
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, truncated , _= env.step(action)
            transtion = [observation,action,reward,next_observation,done]
            replay_buffer.append(transtion)
            observation = next_observation
            score += reward
            steps += 1
            if done or truncated: # or score <0
                print(f"steps {steps} done {done} truncted {truncated}")
                break
        if len(replay_buffer) > batch_size:
            for _ in range(learn_number):
                agent.learn(replay_buffer.sample(batch_size))
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        log_str = f'episode: {episide}, score: {score:.2f} ,avg_score:{average_score:.2f}, alpha:{agent.alpha.numpy():.2f}'
        f.write(log_str+'\n')
        t = int(time.time()-start_time)
        print(log_str + f' time: {t//3600:2}:{t%3600//60:2}:{t%60:2}') 
        if average_score > best_score:
            best_score = average_score
            agent.save_models()
        
finally:
    f.close()
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()
    env.close()
