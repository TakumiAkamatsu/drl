import random
import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
    ) -> None:
        self.capacity = capacity
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
    ) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def make_batch(
        self,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = list(zip(*batch))
        # to Tensor
        state = torch.stack(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.stack(next_state)
        return state, action, reward, next_state