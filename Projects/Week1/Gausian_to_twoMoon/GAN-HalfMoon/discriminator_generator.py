# generate.py
import torch
import matplotlib.pyplot as plt
from generator import Generator

# Load the trained Generator
generator = Generator()
generator.load_state_dict(torch.load("gan_generator_halfmoons.pth"))
generator.eval()

# Generate random Gaussian noise
z = torch.randn(1000, 2)
generated_moons = generator(z).detach().numpy()

# Plot generated moons
plt.scatter(generated_moons[:, 0], generated_moons[:, 1], c='orange', alpha=0.5, label='Generated Moons')
plt.title("Generated Half-Moons")
plt.legend()
plt.show()
