# train.py
import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from generator import Generator
from discriminator import Discriminator

# Dataset: Real half-moons
n_samples = 1000
real_data, _ = make_moons(n_samples=n_samples, noise=0.1)
real_data = torch.tensor(real_data, dtype=torch.float32)
real_dataset = DataLoader(TensorDataset(real_data), batch_size=128, shuffle=True)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers and loss
gen_opt = torch.optim.Adam(generator.parameters(), lr=0.0002)
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = torch.nn.BCELoss()

# Training loop
epochs = 5000
for epoch in range(epochs):
    for real_batch in real_dataset:
        real_samples = real_batch[0]
        batch_size = real_samples.size(0)

        # Training logic: Discriminator
        z = torch.randn((batch_size, 2))
        fake_samples = generator(z).detach()
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))
        disc_real_loss = criterion(discriminator(real_samples), real_labels)
        disc_fake_loss = criterion(discriminator(fake_samples), fake_labels)
        disc_loss = disc_real_loss + disc_fake_loss
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        # Training logic: Generator
        z = torch.randn((batch_size, 2))
        fake_samples = generator(z)
        gen_loss = criterion(discriminator(fake_samples), real_labels)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Disc Loss: {disc_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}")

# Save the trained Generator
torch.save(generator.state_dict(), "gan_generator_halfmoons.pth")
