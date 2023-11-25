# written with reference to https://github.com/jaybutera/tetrisRL/blob/master/dqn_agent.py and https://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import curses
from engine import TetrisEngine
from random import shuffle
from random import sample

use_cuda = torch.cuda.is_available()
if use_cuda:print("....Using Gpu...")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor



# from dqn_agent.py in tetrisrl repo
class BasicFF(nn.Module):
    def __init__(self, fan_in=200, fan_out=7):
        super().__init__()
        #self.bn1 = nn.BatchNorm1d(fan_in)
        self.ff1 = nn.Linear(fan_in, 100)
        self.relu = nn.ReLU()
        #self.bn2 = nn.BatchNorm1d(600)
        self.ff2 = nn.Linear(100, 10)
        #self.bn3 = nn.BatchNorm1d(800)
        self.ff3 = nn.Linear(10, fan_out)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        h = self.relu(self.ff1(x))
        h = self.relu(self.ff2(h))
        return self.ff3(h)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, training_sample):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = training_sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class TAMER:
    def __init__(self, speed):
        self.model = BasicFF()
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        self.history = []
        self.memory = ReplayMemory(3000)
        self.steps = 0
        self.batch_training = 100
        self.batch_size = 32
        self.speed = speed

    @torch.no_grad()
    def select_action(self, state):
        # full greedy action selection
        self.model.eval()
        actions = self.model(Variable(state).type(FloatTensor))
        final_action = actions.data.max(1)[1].view(1, 1)
        self.model.train()
        return final_action
    
    def optimize_supervised(self, pred, targ):
        self.optimizer.zero_grad()
        diff = self.loss(pred, targ)
        diff.backward()
        self.optimizer.step()

    def get_pred_targ(self, state, action, h):
        self.model.eval()
        pred = self.model(Variable(state).type(FloatTensor))
        targ = [[0, 0, 0, 0, 0, 0, 0]]
        targ[0][action] = h
        targ = FloatTensor(targ)
        self.model.train()
        return pred, targ

    def training_step(self, state, action, h):
        self.steps += 1
        # add our state-action pair to the current history (distinct from buffer)
        self.history.append((state.clone(), action))
        # if we have a human reward, add the appropriate states to buffer and optimize model
        if h != 0:
            # set states to apply reward to based on [0.2, 4] second interval
            start = len(self.history) - 3
            end = int(start - (4000 / self.speed))
            if end < 0:
                end = 0
            minibatch = self.history[end:start]
            shuffle(minibatch)
            for state, action in minibatch:
                self.memory.push((state, action, h))
                pred, targ = self.get_pred_targ(state, action, h)
                self.optimize_supervised(pred, targ)
        
        # do a memory batch update if we are at certain step threshold
        if self.steps % self.batch_training == 0 and len(self.memory) >= self.batch_size:
            for state_, action_, h_ in self.memory.sample(self.batch_size):
                pred, targ = self.get_pred_targ(state_, action_, h_)
                self.optimize_supervised(pred, targ)
        


