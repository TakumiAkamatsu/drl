import os
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import gymnasium as gym
import torch
import torch.optim as optim
from tqdm import tqdm
from dqn_agent import DQNAgent
from buffer import ReplayBuffer


class Trainer:
    def __init__(
        self,
        agent: DQNAgent,
        buffer: ReplayBuffer,
        batch_size: int,
        lr: float,
    ) -> None:
        self.agent = agent
        self.buffer = buffer
        self.batch_size = batch_size
        self.lr = lr
        self.env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)

    def train(
        self,
        n_episodes: int,
    ) -> None:
        optimizer = optim.Adam(self.agent.parameters(), lr=self.lr)
        obs, _ = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        for _ in tqdm(range(n_episodes)):
            # explore
            flag = False
            while not flag:
                action = self.agent.act(obs)
                obs_next, reward, done, _, _ = self.env.step(action=action)
                obs_next = torch.from_numpy(obs_next).float().to(self.device)
                self.buffer.push(obs, action, reward, obs_next)
                obs = obs_next
                flag = flag + done
                if len(self.buffer) > self.batch_size:
                    # exploit
                    batch = self.buffer.make_batch(batch_size=self.batch_size)
                    state_batch = batch[0].to(self.device)
                    action_batch = batch[1].to(self.device)
                    reward_batch = batch[2].to(self.device)
                    next_state_batch = batch[3].to(self.device)
                    # train
                    td_difference = self.agent.compute_td_difference(
                        state=state_batch,
                        action=action_batch,
                        reward=reward_batch,
                        next_state=next_state_batch,
                    )
                    loss = td_difference.pow(2).mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        print("Learning has been completed")
        frames = []
        obs, _ = self.env.reset()
        total_reward = 0
        done = False
        # off the epsilon-greedy policy
        self.agent.epsilon = 0
        while not done:
            frames.append(obs)
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.agent.act(obs)
            obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward

        # frames -> video by using ArtistAnimation
        fig = plt.figure()
        ims = []
        for frame in frames:
            im = plt.imshow(frame, animated=True)
            ims.append([im])
        ani = ArtistAnimation(fig, ims, interval=100, blit=True)
        os.makedirs("output", exist_ok=True)
        ani.save("output/video.gif", writer="pillow")