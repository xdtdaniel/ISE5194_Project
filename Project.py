import numpy as np
import gym
from gym.utils.play import *
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# hyperparameters

max_iter = 2000
alpha = 0.0001  # learning rate
epsilon = 0.9999
gamma = 0.9
memory_capacity = 2000
batch_size = 32
target_update = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("MsPacman-ram-v0")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n


exp_read = open("human.txt", mode="rb")
exp = pickle.load(exp_read)
exp_read.close()


def callback(obs_t, obs_tp1, action, rew, done, info):
    if action != 0 and not done:
        exp[tuple(obs_t)] = action


def human_play():
    play(env, callback=callback)
    instruction = open("human.txt", mode="wb")
    pickle.dump(exp, instruction)
    instruction.close()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        state = self.fc1(state)
        # use rectified linear units as activation function
        state = F.relu(state)
        possible_actions = self.out(state)
        return possible_actions


class DQN(object):
    def __init__(self, experience: dict):
        self.evaluate_net = Net()
        self.target_net = Net()
        self.memory = np.zeros((memory_capacity, n_state * 2 + 2))
        self.loss = nn.MSELoss()
        self.step_count = 0
        self.memory_count = 0
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=alpha)
        self.experience = experience

    def choose_action(self, state):
        # 75% of chance choose actions from dictionary
        if tuple(state) in self.experience and np.random.rand() < 0.75 * epsilon:
            action = self.experience[tuple(state)]
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            if np.random.rand() > epsilon:
                possible_actions = self.evaluate_net.forward(state)
                action = torch.max(possible_actions, 1)[1].data.numpy()[0]
            else:
                action = env.action_space.sample()

        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        position = self.memory_count % memory_capacity
        self.memory[position, :] = transition
        self.memory_count += 1

    def learn(self, done):
        if self.step_count % target_update == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.step_count += 1

        random_memory = np.random.choice(memory_capacity, batch_size)
        batch_memory = self.memory[random_memory, :]
        batch_state = torch.FloatTensor(batch_memory[:, :n_state])
        batch_action = torch.LongTensor(batch_memory[:, n_state:n_state + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, n_state + 1:n_state + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -n_state:])

        q_curr = self.evaluate_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward
        if not done:
            q_target += gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss(q_curr, q_target)

        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()


# human_play()

dqn = DQN(exp)
cumulative_reward_through_time = []

print("\nCollecting experience")
total_reward = 0
avg_reward = []
for i in range(max_iter):
    state = env.reset()
    cumulative_reward = 0
    while True:
        env.render()
        action = dqn.choose_action(state)

        next_state, reward, done, info = env.step(action)

        dqn.store_transition(state, action, reward, next_state)

        cumulative_reward += reward
        if dqn.memory_count > memory_capacity:
            dqn.learn(done)

        if done:
            cumulative_reward_through_time.append(cumulative_reward)
            total_reward += cumulative_reward
            print("Ep ", i, "Reward ", cumulative_reward)
            avg_reward.append(total_reward / (i+1))
            epsilon = 0.999 ** i
            break
        state = next_state

plt.plot(avg_reward)
plt.show()
