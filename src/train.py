import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from copy import deepcopy
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), zip(*batch)))

    def __len__(self):
        return len(self.data)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class ProjectAgent:
    def __init__(self):
        self.nb_actions = env.action_space.n
        self.learning_rate = 0.001
        self.gamma = 0.98
        self.buffer_size = 100000
        self.epsilon_min = 0.01
        self.epsilon_max = 1.0
        self.epsilon_decay_period = 20000
        self.epsilon_delay = 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        self.batch_size = 800
        self.gradient_steps = 5
        self.update_target_strategy = 'replace'
        self.update_target_freq = 900
        self.criterion = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.memory = ReplayBuffer(self.buffer_size, self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def act(self, observation):
        with torch.no_grad():
            q_values = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(q_values).item()

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "final_dqn.pt"))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "final_dqn.pt"), map_location=self.device))
        self.model.eval()

    def train(self, max_episode):
        epsilon = self.epsilon_max
        global_step = 0
        episode_returns = []

        for episode in tqdm(range(max_episode), desc="Training Progress"):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                if global_step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_reward += reward

                for _ in range(self.gradient_steps):
                    if len(self.memory) >= self.batch_size:
                        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                        max_next_q_values = self.target_model(next_states).max(1)[0].detach()
                        target = rewards + self.gamma * max_next_q_values * (1 - dones)
                        q_values = self.model(states).gather(1, actions.long().unsqueeze(1)).squeeze()
                        loss = self.criterion(q_values, target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                if self.update_target_strategy == 'replace' and global_step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                state = next_state
                global_step += 1

            episode_returns.append(episode_reward)

            tqdm.write(f"Episode {episode + 1}/{max_episode} | "
                       f"Epsilon {epsilon:.3f} | "
                       f"Reward {episode_reward:.2f}")

        self.save(os.getcwd())
        return episode_returns

if __name__ == "__main__":
    torch.manual_seed(42)
    agent = ProjectAgent()
    agent.train(max_episode=500)
