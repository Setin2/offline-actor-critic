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
import matplotlib.pyplot as plt
import scipy.stats
import copy

EPISODES = 200 
REPLAY_BUFFER_SIZE = 10000 # equivalent to 4 episodes
GAMMA = 0.8  # Q-learning discount factor
LR = 1e-4# NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 256  # Q-learning batch size
STATE_DIM = 84*84*3#48*48*3
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
        state = state.view(state.size(0), -1)
        x = torch.cat([state, action], 1)
        action_value = F.relu(self.linear1(x))
        action_value = F.relu(self.linear2(action_value))
        action_value = self.q1(action_value) 
        return action_value

class ActorNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(STATE_DIM, HIDDEN_LAYER)
        self.linear2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        
        self.mu = nn.Linear(HIDDEN_LAYER, ACTION_DIM)
        self.log_var = nn.Linear(HIDDEN_LAYER, ACTION_DIM)
       
        self.optimizer = optim.Adam(self.parameters(), LR)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        return mu, log_var

class Agent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        #self.load_models()
        
    # load previoulsy trained policy + its optimizer if there is any. can resume training from last checkpoint if needed
    def load_models(self):
        if os.path.exists("model_weights.pth"):
            print("LOADING POLICY")
            checkpoint = torch.load('model_weights.pth')
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.actor.eval()

            checkpoint = torch.load('critic_model_weights.pth')            
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.critic.eval()
            
    def sample_action(self, state):
        state = state.view(state.size(0), -1)
        mu, log_var = self.actor.forward(state)
        
        normal = Normal(0, 1)
        epsilon = normal.sample()
        action = mu + torch.exp(log_var).sqrt() * epsilon
        
        action[:,0] = torch.sigmoid(action[:,0])
        action[:,1] = torch.tanh(action[:,1])
        action[:,2] = torch.sigmoid(action[:,2])
        
        log_prob = -0.5*log_var - (0.5/(log_var+0.001)) * (action - mu)**2 - 0.5*math.log(2*np.pi)

        return action, log_prob

    def sample_behavioural_action(self, state):
        state = state.view(state.size(0), -1)
        mu, log_var = self.actor.forward(state)
        
        normal = Normal(0, 1)
        epsilon = normal.sample()
        predicted_action = mu + torch.exp(log_var).sqrt() * epsilon

        return predicted_action

    def select_action(self, observation):
        state = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        action = self.sample_action(state)#self.sample_behavioural_action(state)
        return action[0].detach().numpy()[0].tolist()

    def populate_replay_buffer(self, num_steps):
        print('Loading offline data')
        with open('dataset_obs.p', 'rb') as states, open('dataset_actions.p', 'rb') as actions, open('dataset_rewards.p', 'rb') as rewards:
            for i in range(num_steps):
                self.memory.push((pickle.load(states)/255.0, pickle.load(actions), pickle.load(rewards)))
        print('Data loaded')

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE + 1)
        batch_state, batch_action, batch_reward = zip(*transitions)
        
        state = torch.FloatTensor(np.array(batch_state[0 : len(batch_state) - 1])).to(self.device)
        action = torch.FloatTensor(np.array(batch_action[0 : len(batch_action) - 1])).to(self.device)
        reward = torch.FloatTensor(np.array(batch_reward[0 : len(batch_reward) - 1])).to(self.device)
        new_state = torch.FloatTensor(np.array(batch_state[1 : len(batch_state)])).to(self.device)

        self.critic_learn(state, action, reward, new_state)
        self.actor_learn(state, new_state)
        #self.behavioural_cloning(state, action)

    def critic_learn(self, obs, actions, rewards, next_obs):
        next_action, _ = self.sample_action(next_obs)
        max_next_q_values = self.critic(next_obs, next_action)
        expected_q_values = rewards + GAMMA * max_next_q_values.view(-1)
        predicted_q_values = self.critic(obs, actions)

        critic_loss = F.smooth_l1_loss(predicted_q_values, expected_q_values)

        critic_loss_list.append(critic_loss) # for plotting loss

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def behavioural_cloning(self, obs, action):
        sample_action = self.sample_behavioural_action(obs)

        criterion = nn.MSELoss()
        actor_loss = criterion(action, sample_action)

        #actor_loss_list.append(actor_loss) # for plotting loss

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def actor_learn(self, obs, next_obs):
        new_action, _ = self.sample_action(obs)
        q1 = self.critic(obs, new_action)

        new_action2, log_prob = self.sample_action(next_obs)
        expected_q = self.critic(next_obs, new_action2)

        A_hat = q1 - expected_q

        actor_loss = -torch.mean(log_prob*A_hat)
        
        actor_loss_list.append(actor_loss) # for plotting loss

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
    
    def train(self, epochs, plot):
        self.actor.train()
        self.critic.train()

        for i in range(epochs):
            self.learn()
            if plot: plot_loss(ax)
            
        plt.savefig('books_read.png')
    
        self.save_models()

        self.actor.eval()
        self.critic.eval()

    def save_models(self):
        print("TRAINING FINISHED, SAVING MODELS")

        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor.optimizer.state_dict()
            }, "model_weights.pth")
            
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic.optimizer.state_dict()
            }, "critic_model_weights.pth")

def plot_loss(ax):
    loss_list_t = torch.FloatTensor(critic_loss_list)
    ax[0].plot(loss_list_t.numpy(), 'k')
    loss_list_t = torch.FloatTensor(actor_loss_list)
    ax[1].plot(loss_list_t.numpy(), 'k')
    ax[2].plot(RMSELIST, 'k')    
    plt.pause(0.01)

# use test set to compute RMSE
def test(num_steps):
    RMSEL = []

    transitions = buffer_test.sample(num_steps)
    for s, a, r, in transitions:
        predicted_action = agent.select_action(s)
        RMSEL.append(math.sqrt(np.square(np.subtract(a, predicted_action)).mean()))      

    RMSEL_LEN = len(RMSEL)
    RMSE = sum(RMSEL) / RMSEL_LEN
    RMSELIST.append(RMSE)

if __name__ == '__main__':
    actor_loss_list = []
    critic_loss_list = []
    RMSELIST = []

    agent = Agent()
    
    agent.populate_replay_buffer(REPLAY_BUFFER_SIZE)

    buffer_test = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    fig, ax = plt.subplots(3)
    ax[0].title.set_text('Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax[1].title.set_text('Actor Loss')
    ax[2].title.set_text('RMSE')

    with open('dataset_obs.p', 'rb') as states, open('dataset_actions.p', 'rb') as actions, open('dataset_rewards.p', 'rb') as rewards:
        for i in range(1000):
            buffer_test.push((pickle.load(states)/255.0, pickle.load(actions), pickle.load(rewards)))

    agent.train(50, plot=True)

    """# for running the policy in the carla environment
    env = env.CarEnv()
    print("STARTING FIRST EPISODE")
    for e in range(EPISODES):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            act = agent.select_action(obs)
            print(act)
            new_state, reward, done, info = env.step(act)
            steps += 1
            obs = new_state
            if done:
                print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                for actor in env.actor_list:
                    actor.destroy()
                
                break"""
