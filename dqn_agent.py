import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DQNAgent(nn.Module):
    def __init__(
        self,
        discount_factor: float,
        seed: int,
    ):
        super(DQNAgent, self).__init__()
        self.epsilon = 0.1
        self.discount_factor = discount_factor
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=8,
            stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=4,
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        # discrete daction : (do nothing, left, right, gas, brake)
        self.out = nn.Linear(in_features=4096, out_features=5)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out = self.conv1(x.permute(0, 3, 1, 2))
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = torch.flatten(out, start_dim=1)
        out = self.out(out)
        return out
    
    def act(
        self,
        state: Tensor,
    ) -> int:
        # epsilon-greedy
        if torch.rand(1) > self.epsilon:
            with torch.no_grad():
                return torch.argmax(self.forward(state)).item()
        else:
            return torch.randint(low=0, high=5, size=(1,)).item()
        
    def compute_td_difference(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        td_target = reward + self.discount_factor * torch.max(self.forward(next_state))
        td_current = self.forward(state)[range(0, action.numel()), action]
        return td_target - td_current