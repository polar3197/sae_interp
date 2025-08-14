import torch
import torch.nn as nn
from typing import Tuple

class SparseAutoencoder(nn.Module):
    """
    A simple Sparse Autoencoder model with a single hidden layer.

    Args:
        input_size (int): Dimensionality of the input features.
        hidden_size (int): Dimensionality of the hidden (encoded) layer.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor, expected shape (batch_size, input_size).
                              Note: Flattening is expected to happen *before* this call
                              if the original input is not flat (e.g., images).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - reconstructed_x (torch.Tensor): The output of the decoder.
                - encoded_z (torch.Tensor): The output of the encoder (latent representation).
        """
        # Assume input x is already flattened if necessary
        encoded_z = self.encoder(x)
        reconstructed_x = self.decoder(encoded_z)
        return reconstructed_x, encoded_z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation."""
        z = self.decoder(z)
        return z
