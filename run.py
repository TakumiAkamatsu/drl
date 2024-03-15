import argparse
from buffer import ReplayBuffer
from dqn_agent import DQNAgent
from trainer import Trainer


def main(
    n_episodes: int,
    capacity: int,
    batch_size: int,
    lr: float,
    discount_factor: float,
):
    agent = DQNAgent(
        discount_factor=discount_factor,
        seed=0,
    )
    replay_buffer = ReplayBuffer(
        capacity=capacity,
    )
    trainer = Trainer(
        agent=agent,
        buffer=replay_buffer,
        batch_size=batch_size,
        lr=lr,
    )
    trainer.train(n_episodes=n_episodes)


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100000)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    args = parser.parse_args()
    main(
        n_episodes=args.n_episodes,
        capacity=args.capacity,
        batch_size=args.batch_size,
        lr=args.lr,
        discount_factor=args.discount_factor,
    )