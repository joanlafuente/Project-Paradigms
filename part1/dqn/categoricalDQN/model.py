import torch.nn as nn
import torch.nn.functional as F
import torch

class C51DQN(nn.Module):
    def __init__(self, state_shape, n_actions, n_atoms, v_min, v_max):
        super(C51DQN, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.z_values = torch.linspace(v_min, v_max, n_atoms)

        # Neural network architecture
        self.conv1 = nn.Conv2d(in_channels=state_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # New line to ensure the output is 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions * n_atoms)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        print(f"Conv1 shape: {x.shape}")
        x = F.relu(self.conv2(x))
        print(f"Conv2 shape: {x.shape}")
        x = F.relu(self.conv3(x))
        print(f"Conv3 shape: {x.shape}")
        self.adaptive_pool(x)
        print(f"Apadtive shape: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"Flatten shape: {x.shape}")
        x = F.relu(self.fc1(x))
        print(f"FC1 shape: {x.shape}")
        x = self.fc2(x)
        print(f"FC2 shape: {x.shape}")
        x = x.view(-1, self.n_actions, self.n_atoms)
        x = F.softmax(x, dim=-1)  # Normalize the distribution over atoms
        return x
