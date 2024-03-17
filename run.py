import argparse

from buffer import ReplayBuffer
from dqn_agent import DQNAgent
from trainer import Trainer


def main(
    n_episodes: int,
    update_interval: int,
    capacity: int,
    batch_size: int,
    lr: float,
    discount_factor: float,
):
    agent = DQNAgent(
        discount_factor=discount_factor,
        seed=0,
        lr=lr,
    )
    replay_buffer = ReplayBuffer(
        capacity=capacity,
    )
    trainer = Trainer(
        agent=agent,
        buffer=replay_buffer,
        batch_size=batch_size,
    )
    trainer.train(n_episodes=n_episodes, update_interval=update_interval)


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100000)
    parser.add_argument("--update_interval", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    args = parser.parse_args()
    main(
        n_episodes=args.n_episodes,
        update_interval=args.update_interval,
        capacity=args.capacity,
        batch_size=args.batch_size,
        lr=args.lr,
        discount_factor=args.discount_factor,
    )