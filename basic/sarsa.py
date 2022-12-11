import numpy as np
import os
import time

class Agent(object):
    
    def __init__(self,alpha=0.001,gamma=0.8,e_greedy=0.9) -> None:
        
        self.alpha=alpha 
        self.gamma=gamma
        self.e_greedy=e_greedy
        self.q_table=np.zeros((16,4))

    def choose_action(self,state):
        if (np.random.uniform() > self.e_greedy) or (self.q_table.any() == False): 
            action=np.random.choice([0,1,2,3])
        else:
            action=self.q_table[state,:].argmax()
        return action
    
    def sarsa_learn(self,s,a,r,s_,a_):
        q_predict = self.q_table[s,a]
        if s_ != 15:
            q_target = r + self.gamma*self.q_table[s_,a_]
        else:
            q_target = r
        self.q_table[s,a]+=self.alpha*(q_target-q_predict)
        
class Environment(object):
    def __init__(self):
        self.map =np.zeros((4,4),int)
        self.map[1,1]=8
        self.map[2,2]=8
        self.done = False
        self.up_array=np.array([4,5,6,7,8,9,10,11,12,13,14,15])
        self.down_array=np.array([0,1,2,3,4,5,6,7,8,9,10,11])
        self.left_array = np.array([1,2,3,5,6,7,9,10,11,13,14,15])
        self.right_array = np.array([0,1,2,4,5,6,8,9,10,12,13,14])
        
        self.state=0

    def step(self,state,action):

        if action == 0:
            new_state = state-4 if np.any(state==self.up_array) else state
        if action == 1:
            new_state = state+4 if np.any(state==self.down_array) else state
        if action == 2:
            new_state = state-1 if np.any(state==self.left_array) else state
        if action == 3:
            new_state = state+1 if np.any(state==self.right_array) else state
        reward = -1
        if new_state==5:
            reward=-100
            self.done = True
        if new_state==10:
            reward=-100
            self.done = True
        if new_state==15:
            reward=100
            self.done = True
        self.state=new_state
        return reward,new_state

    def reset(self):
        self.done=False
        self.state=0
    
    def render(self):
        i_row=self.state//4
        i_col=self.state%4
        new_map=self.map.copy()
        new_map[i_row][i_col]=7
        print(new_map)
        time.sleep(0.3)
        os.system('clear')

def main():
    
    agent = Agent()
    env   = Environment()
    for _ in range(1000):
        step_counter = 0
        s = 0
        a = agent.choose_action(s)
        env.reset()
        s_list = [0]
        while not env.done:
            env.render()
            r,s_ = env.step(s,a)
            a_ = agent.choose_action(s_)
            agent.sarsa_learn(s,a,r,s_,a_)
            s=s_
            a=a_
            step_counter+=1
            s_list.append(s)
        
        print(step_counter)
        print(s_list)
        time.sleep(0.3)
    
if __name__== '__main__':
    main()