import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

class CarRacingDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open('observation.pickle','rb') as f:
            self.x_data = pickle.load(f)
        with open('action.pickle','rb') as f:
            self.y_data = pickle.load(f)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        x = np.transpose(self.x_data[index],(2,0,1)).astype('float32')
        y = self.y_data[index].astype('float32')
        return x,y

    def __len__(self):
        return self.len

dataset = CarRacingDataset()

train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2)

class ConvNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.cnn = nn.Sequential(  
            nn.Conv2d(4 , 8, kernel_size=4, stride=2),# input (4, 84, 84)
            nn.ReLU(),  
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # input (8, 41, 41)
            nn.ReLU(),  
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # input  (16, 20, 20)
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # input (32, 9, 9)
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=4, stride=1),  # input (64, 4, 4)
            nn.ReLU(),  
        )  # output shape (128, 1, 1)
    
    def forward(self,x):
        x = self.cnn(x)
        return x

class Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = ConvNN()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,128)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x 

policy = Policy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(policy.parameters(),lr=1e-3)

def train(epoch):
    running_loss = 0.0
    for idx, data in enumerate(train_loader,0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        y_pred = policy(inputs)
        loss = criterion(y_pred,targets)
        running_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 99:
            print(f'epoch {epoch} idx {idx} loss {running_loss / 100 :.4f} ')
            running_loss = 0.0

def test():
    import gym
    import numpy as np
    import matplotlib.pyplot as plt
    from wrapper import CarWrapper
    
    env = CarWrapper(gym.make('CarRacing-v2',render_mode='human'))
    score_list = []
    average_score_list = []
    total_eposide = 1
    for episide in range(total_eposide):
        steps = 0
        score = 0
        observation,_ = env.reset()
        while True:
            obs = torch.tensor(observation.transpose((2,0,1)).astype('float32'))
            with torch.no_grad():
                action = policy(obs)
            next_observation, reward, done, truncated , _ = env.step(action[0].numpy())
            observation = next_observation
            score += reward
            steps+=1
            env.render()
            if done or truncated:
                break
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        print(  'episode %d/%d'% (episide,total_eposide),
                'score %.1f' % score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--path",  default='./policy.pth')
    parser.add_argument("--train", default='test')
    args = parser.parse_args()

    for epoch in range(args.epoch):
        if args.train == 'train':
            train(epoch)
            torch.save(policy.state_dict(),args.path)
        else:
            policy.load_state_dict(torch.load(args.path))
            policy.to("cpu")
            test()
            pass
