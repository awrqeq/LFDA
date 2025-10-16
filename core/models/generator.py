import torch
import torch.nn as nn


class TriggerNet(nn.Module):
    """
    A small CNN that acts as the trigger.
    It takes a frequency patch as input and outputs a perturbation patch.
    This architecture is designed to be lightweight and effective.
    """

    def __init__(self, in_channels=2, out_channels=2):
        super(TriggerNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input frequency patch to generate a perturbation patch.
        Args:
            x (torch.Tensor): The input frequency patch, with shape (B, 2, H, W)
                              where 2 channels are real and imaginary parts.
        Returns:
            torch.Tensor: The output perturbation patch of the same shape.
        """
        perturbation = self.layers(x)
        # Use tanh to bound the output to [-1, 1], which will be scaled by injection_strength later
        return torch.tanh(perturbation)