# discriminator.py
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),    # Input: Coordinates
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),    # Output: Probability (Real or Fake)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
