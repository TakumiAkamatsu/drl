from dqn_agent import DQNAgent
from buffer import ReplayBuffer
from trainer import Trainer


def main():
    agent = DQNAgent(
        discount_factor=0.99,
        seed=0,
    )
    replay_buffer = ReplayBuffer(
        capacity=1000,
    )
    trainer = Trainer(
        agent=agent,
        buffer=replay_buffer,
        batch_size=32,
        lr=1e-3,
    )
    trainer.train(n_episodes=1000)


if __name__ == '__main__':
    main()