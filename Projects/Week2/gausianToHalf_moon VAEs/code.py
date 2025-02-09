import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate half-moon data
X, _ = make_moons(n_samples=1000, noise=0.1)
X = torch.FloatTensor(X)

# Create DataLoader
dataloader = DataLoader(X, batch_size=64, shuffle=True)

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize VAE and optimizer
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate samples
vae.eval()
with torch.no_grad():
    z = torch.randn(1000, 2)
    generated_moons = vae.decode(z).numpy()

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Original')
plt.title('Original Half-Moons')
plt.legend()

plt.subplot(122)
plt.scatter(generated_moons[:, 0], generated_moons[:, 1], c='red', alpha=0.5, label='Generated')
plt.title('VAE Generated Half-Moons')
plt.legend()

plt.tight_layout()
plt.show()
