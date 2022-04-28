import pickle
import numpy as np
import argparse
import datetime
import glob
import os
import pickle
import random
import sys
import time
import env
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Normal

EPISODES = 200 # number of episodes
REPLAY_BUFFER_SIZE = 10000 # equivalent to 1 episodes 
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size
STATE_DIM = 48*48*3
ACTION_DIM = 3

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.linear1 = nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_LAYER)
        self.linear2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.q1 = nn.Linear(HIDDEN_LAYER, 1)
        self.optimizer = optim.Adam(self.parameters(), LR)

    def forward(self, state, action):
        out = state.view(state.size(0), -1)
        action_value = self.linear1(torch.cat([out, action], dim=1)) 
        action_value = F.relu(action_value) 
        action_value = self.linear2(action_value) 
        action_value = F.relu(action_value) 
        q1 = self.q1(action_value) 
        return q1

class ActorNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(STATE_DIM, HIDDEN_LAYER)
        self.linear2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.linear3 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.optimizer = optim.Adam(self.parameters(), LR)
        
        self.mu = nn.Linear(HIDDEN_LAYER, ACTION_DIM)
        self.var = nn.Linear(HIDDEN_LAYER, ACTION_DIM)

    def forward(self, obs):
        x = obs.view(obs.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        
        mu = self.mu(x)
        std = self.var(x)
        std = torch.clamp(std, min=1e-6, max=1)
        
        return mu, std
        
    def sample(self, state):
        x = state.view(state.size(0), -1)
        mu, std = self.forward(state)
        normal = Normal(0, 1)
        epsilon = normal.sample()
        action = mu + std * epsilon

        log_prob = normal.log_prob(epsilon)
        
        return action, log_prob

class Agent(object):
    def __init__(self):
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()

    def select_action(self, observation):
        state = Variable(FloatTensor(np.array([observation])))
        action, _ = self.actor.sample(state)
        return action.detach().numpy()[0].tolist()

    def populate_replay_buffer(self, states_file, actions_file, rewards_file, num_steps):
        print('Loading offline data')
        for i in range(num_steps):
            self.memory.push((FloatTensor(np.array([pickle.load(states_file)])), FloatTensor(np.array([pickle.load(actions_file)])), FloatTensor(np.array([pickle.load(rewards_file)]))))
        print('Data loaded')

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE + 1)
        batch_state, batch_action, batch_reward = zip(*transitions)
        
        state = Variable(torch.cat(batch_state[0 : len(batch_state) - 1]))
        action = Variable(torch.cat(batch_action[0 : len(batch_action) - 1]))
        reward = Variable(torch.cat(batch_reward[0 : len(batch_reward) - 1]))
        new_state = Variable(torch.cat(batch_state[1 : len(batch_state)]))

        self.critic_learn(state, action, reward, new_state)
        self.actor_learn(state)

    def critic_learn(self, obs, action, reward, next_obs):
        next_action, _ = self.actor.sample(next_obs)
        max_next_q_values = self.critic(next_obs, next_action)
        expected_q_values = reward + GAMMA * max_next_q_values
        predicted_q_values = self.critic(obs, action)

        critic_loss = F.smooth_l1_loss(predicted_q_values, expected_q_values)
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def actor_learn(self, obs):
        new_action, _ = self.actor.sample(obs)
        q1 = self.critic(obs, new_action)
        
        new_action2, log_prob = self.actor.sample(obs)
        expected_q = self.critic(obs, new_action2)
        
        A_hat = q1 - expected_q
        
        #actor_loss = 
        
        #self.actor.optimizer.zero_grad()
        #actor_loss.backward()
        #self.actor.optimizer.step()

if __name__ == '__main__':
    states = open('dataset_obs.p', 'rb')
    actions = open('dataset_actions.p', 'rb')
    rewards = open('dataset_rewards.p', 'rb')

    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    env = env.CarEnv()
    agent = Agent()
    agent.populate_replay_buffer(states, actions, rewards, REPLAY_BUFFER_SIZE)

    for e in range(EPISODES):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            act = agent.select_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.learn()
            steps += 1
            obs = new_state

            if done:
                print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                for actor in env.actor_list:
                    actor.destroy()
                
                break
















