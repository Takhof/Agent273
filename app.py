import random
import argparse
import numpy as np
from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import imageio

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size=10000,
                 batch_size=64,
                 gamma=0.99,
                 lr=1e-3,
                 tau=1e-3,
                 update_every=4,
                 device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.buffer) >= self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, eps=0.0):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = [x.to(self.device) for x in (states, actions, rewards, next_states, dones)]
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        for t_param, l_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max-t', type=int, default=1000)
    parser.add_argument('--record-gif', type=str, default=None,
                        help='Path to save a demo GIF after training')
    args = parser.parse_args()

    env = gym.make(args.env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    scores, eps = [], 1.0
    for i in range(1, args.episodes+1):
        reset = env.reset()
        state = reset[0] if isinstance(reset, tuple) else reset
        score = 0
        for _ in range(args.max_t):
            action = agent.act(state, eps)
            step = env.step(action)
            if len(step) == 5:
                next_state, reward, terminated, truncated, _ = step; done = terminated or truncated
            else:
                next_state, reward, done, _ = step
            agent.step(state, action, reward, next_state, done)
            state, score = next_state, score + reward
            if done: break
        scores.append(score)
        eps = max(0.01, 0.995 * eps)
        if i % 10 == 0:
            print(f"{args.env} Episode {i}\tAvg Score: {np.mean(scores[-10:]):.2f}\tEps: {eps:.2f}")

    env.close()

    # Record GIF if requested
    if args.record_gif:
        print(f"Recording GIF to {args.record_gif} ...")
        env_gif = gym.make(args.env, render_mode='rgb_array')
        frames = []
        reset = env_gif.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        done = False
        while not done:
            frames.append(env_gif.render())
            action = agent.act(obs, 0.0)
            step = env_gif.step(action)
            if len(step) == 5:
                obs, _, terminated, truncated, _ = step; done = terminated or truncated
            else:
                obs, _, done, _ = step
        env_gif.close()
        imageio.mimsave(args.record_gif, frames, fps=30)
        print(f"Saved GIF at {args.record_gif}")
