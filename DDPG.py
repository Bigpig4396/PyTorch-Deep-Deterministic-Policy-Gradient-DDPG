import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x * 2   # only for this environment
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim)
        self.p_net = PolicyNetwork(state_dim, action_dim)
        self.q_criterion = nn.MSELoss()
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), lr=3e-4)
        self.gamma = 0.9

    def get_action(self, state, epsilon):
        a = self.p_net(torch.from_numpy(state).float())
        a = a + epsilon * torch.randn(self.action_dim)
        a = torch.clamp(a, min=-2, max=2)
        return a.detach().numpy()

    def train(self, batch):
        state = batch[0]    # array [64 1 2]
        action = batch[1]   # array [64, ]
        reward = batch[2]   # array [64, ]
        next_state = batch[3]
        done = batch[4]

        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float().view(-1, self.action_dim)
        next_state = torch.from_numpy(next_state).float()
        next_action = self.p_net(next_state)
        reward = torch.FloatTensor(reward).float().unsqueeze(1)


        q = self.q_net(state, action)
        next_q = self.q_net(next_state, next_action)
        est_q = reward + self.gamma * next_q

        q_loss = self.q_criterion(q, est_q.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        p_loss = -self.q_net(state, self.p_net(state)).mean()
        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim, action_dim)
    max_epi_iter = 100
    max_MC_iter = 500
    batch_size = 64
    replay_buffer = ReplayBuffer(50000)
    train_curve = []
    for epi in range(max_epi_iter):
        state = env.reset()
        acc_reward = 0
        for MC_iter in range(max_MC_iter):
            # print("MC= ", MC_iter)
            env.render()
            action1 = agent.get_action(state, 1.0-(epi/max_epi_iter))
            next_state, reward, done, info = env.step(action1)
            acc_reward = acc_reward + reward
            replay_buffer.push(state, action1, reward, next_state, done)
            state = next_state
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer.sample(batch_size))
            if done:
                break
        print('Episode', epi, 'reward', acc_reward)
        train_curve.append(acc_reward)
    plt.plot(train_curve, linewidth=1, label='DDPG')
    plt.show()
