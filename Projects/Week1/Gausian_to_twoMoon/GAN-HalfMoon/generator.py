# generator.py
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),  # Input: Gaussian noise
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)   # Output: Half-moon coordinates
        )
    
    def forward(self, z):
        return self.net(z)
