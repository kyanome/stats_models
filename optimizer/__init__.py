from .base import Optimizer
from .vae_optimizer import VAE_Optimizer

optimizer_list = {
    "VAE_Optimizer" : VAE_Optimizer,
}