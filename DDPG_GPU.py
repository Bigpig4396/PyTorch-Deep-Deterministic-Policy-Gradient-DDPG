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
        self.fc1 = nn.Linear(num_inputs + num_actions, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        # print(state.size())
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_1_net = QNetwork(state_dim, action_dim).to(device)
        self.q_2_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_1_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_2_net = QNetwork(state_dim, action_dim).to(device)
        for target_param, param in zip(self.target_q_1_net.parameters(), self.q_1_net.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_q_2_net.parameters(), self.q_2_net.parameters()):
            target_param.data.copy_(param.data)
        self.soft_tau = 1e-2

        self.p_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.q_criterion = nn.MSELoss()
        self.q_1_optimizer = optim.Adam(self.q_1_net.parameters(), lr=1e-3)
        self.q_2_optimizer = optim.Adam(self.q_2_net.parameters(), lr=1e-3)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), lr=3e-4)
        self.gamma = 0.99

    def get_action(self, state, epsilon):
        a = self.p_net.forward(torch.from_numpy(state).float().to(device))
        a = a + epsilon * torch.randn(self.action_dim)
        a = torch.clamp(a, min=-1, max=1)
        return a.detach().cpu().numpy()

    def train(self, batch):
        state = batch[0]    # array [64 1 2]
        action = batch[1]   # array [64, ]
        reward = batch[2]   # array [64, ]
        next_state = batch[3]
        done = batch[4]

        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().view(-1, self.action_dim).to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        next_action = self.p_net.forward(next_state).to(device)
        reward = torch.FloatTensor(reward).float().unsqueeze(1).to(device)

        q1 = self.q_1_net.forward(state, action)
        q2 = self.q_2_net.forward(state, action)
        # q_min = torch.min(q1, q2)
        next_q1 = self.target_q_1_net.forward(next_state, next_action)
        next_q2 = self.target_q_2_net.forward(next_state, next_action)
        next_q_min = torch.min(next_q1, next_q2)
        est_q = reward + self.gamma * next_q_min

        q_loss = self.q_criterion(q1, est_q.detach())
        self.q_1_optimizer.zero_grad()
        q_loss.backward()
        self.q_1_optimizer.step()
        q_loss = self.q_criterion(q2, est_q.detach())
        self.q_2_optimizer.zero_grad()
        q_loss.backward()
        self.q_2_optimizer.step()

        new_a = self.p_net.forward(state)
        q1 = self.q_1_net.forward(state, new_a)
        q2 = self.q_2_net.forward(state, new_a)
        q_min = torch.min(q2, q1)
        p_loss = -q_min.mean()
        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

        for target_param, param in zip(self.target_q_1_net.parameters(), self.q_1_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_q_2_net.parameters(), self.q_2_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def load_model(self):
        print('load model')
        self.q_1_net = torch.load('DDPG_q_net1.pkl').to(device)
        self.q_2_net = torch.load('DDPG_q_net2.pkl').to(device)
        self.target_q_1_net = torch.load('DDPG_target_q_net1.pkl').to(device)
        self.target_q_2_net = torch.load('DDPG_target_q_net2.pkl').to(device)
        self.p_net = torch.load('DDPG_policy_net.pkl').to(device)

    def save_model(self):
        torch.save(self.q_1_net, 'DDPG_q_net1.pkl')
        torch.save(self.q_2_net, 'DDPG_q_net2.pkl')
        torch.save(self.target_q_1_net, 'DDPG_target_q_net1.pkl')
        torch.save(self.target_q_2_net, 'DDPG_target_q_net2.pkl')
        torch.save(self.p_net, 'DDPG_policy_net.pkl')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('state size:', env.observation_space.shape)
    print('action size:', env.action_space.shape)
    agent = DDPG(state_dim, action_dim)
    agent.load_model()
    max_epi_iter = 100
    max_MC_iter = 200
    batch_size = 64
    replay_buffer = ReplayBuffer(50000)
    train_curve = []
    for epi in range(max_epi_iter):
        state = env.reset()
        acc_reward = 0
        for MC_iter in range(max_MC_iter):
            # print("MC= ", MC_iter)
            env.render()
            # action1 = agent.get_action(state, 1.0-(epi/max_epi_iter))
            action1 = agent.get_action(state, 0.0)
            next_state, reward, done, info = env.step(action1)
            acc_reward = acc_reward + reward
            replay_buffer.push(state, action1, reward, next_state, done)
            state = next_state
            if len(replay_buffer) > batch_size:
                # print('train')
                agent.train(replay_buffer.sample(batch_size))
            if done:
                break
        print('Episode', epi, 'reward', acc_reward)
        train_curve.append(acc_reward)
        if epi % 50 == 0:
            agent.save_model()
    plt.plot(train_curve, linewidth=1, label='DDPG')
    plt.show()