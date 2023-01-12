import gym
from agent import Agent
from model import CarWrapper
import pickle
import os


try:
    env = CarWrapper(gym.make('CarRacing-v2'))#,render_mode='human'))
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(action_dim=action_dim)
    # load model
    agent.load_models()
    total_eposide = 20
    
    inputs = []
    labels = []
    for episide in range(total_eposide):
        steps = 0
        score = 0
        observation,_ = env.reset()
        while True:
            action = agent.predict(observation).numpy()
            inputs.append(observation)
            labels.append(action)
            next_observation, reward, done, truncated , _ = env.step(action)
            observation = next_observation
            score += reward
            steps+=1
            #env.render()
            if done or truncated:
                print(f"done {done} truncted {truncated} steps {steps}")
                break
        print(  'episode %d/%d'% (episide,total_eposide),
                'score %.1f' % score)
    
    with open('observation.pickle','wb') as f:
        pickle.dump(inputs,f)
    with open('action.pickle','wb') as ff:
        pickle.dump(labels,ff)
        
finally:
    env.close()