import torch
from torch import nn
from torch.nn import functional as F
from utils.types_ import *
from .base import BaseVAE

class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: int,
                 latent_dim: int,
                 ) -> None:
        super(VanillaVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims, latent_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, in_channels),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        mu, logvar = torch.chunk(result, 2, dim=1)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)
     
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> List[Tensor]:
        input = input.view(input.shape[0], -1)
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), input, mu, logvar]

    def loss_function(self, *args) -> float:
        """
        Computes the VAE loss function.
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        logvar = args[3]
        BCE = F.binary_cross_entropy(recons, input, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return BCE + KLD
