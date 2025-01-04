import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from copy import deepcopy

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Create the environment
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

# Define the ReplayBuffer
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


# Define the DQN model
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


# Define the agent
class ProjectAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(100000, self.device)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_delay = 20
        self.update_target_freq = 100
        self.best_score = float('-inf')

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(env.action_space.n)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_values = self.model(states).gather(1, actions.long().unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criterion(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def train(self, episodes):
        global_step = 0
        for episode in range(episodes):
            state = env.reset()[0]
            total_reward = 0
            done = False
            while not done:
                # Epsilon decay with delay
                if global_step > self.epsilon_delay:
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                action = self.act(state)
                next_state, reward, done, _, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                self.replay()
                global_step += 1

                # Update the target network periodically
                if global_step % self.update_target_freq == 0:
                    self.update_target_model()

            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

            # Save the best model
            if total_reward > self.best_score:
                self.best_score = total_reward
                self.save("best_dqn_model.pt")
                print(f"New best model saved with reward: {total_reward:.2f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ProjectAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, device=device)
    agent.train(episodes=1000)
