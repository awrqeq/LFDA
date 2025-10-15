import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class AssociativeLoss(nn.Module):
    """
    Calculates the conditional cross-entropy loss to associate the trigger
    with the target class. The loss is only active for samples belonging
    to the target class.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, target_class: int) -> torch.Tensor:
        # Create a mask to select only the samples from the target class
        mask = (targets == target_class)

        # If no samples from the target class are in the batch, return zero loss
        if not mask.any():
            return torch.tensor(0.0, device=outputs.device)

        # Filter the outputs and create corresponding targets for the loss calculation
        target_outputs = outputs[mask]
        attack_targets = torch.full_like(targets[mask], target_class)

        return self.criterion(target_outputs, attack_targets)


class LatentLoss(nn.Module):
    """
    (Corrected Implementation)
    Calculates the negative mean squared error between two feature representations.
    The goal is to MAXIMIZE the L2 distance between the features of the poisoned
    image and the clean image in the latent space. By minimizing the negative
    MSE, we effectively maximize this distance, forcing feature disentanglement.
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, h_p: torch.Tensor, h_x: torch.Tensor) -> torch.Tensor:
        # We want to MAXIMIZE the MSE (squared L2 distance).
        # The optimizer MINIMIZES the loss, so we return the NEGATIVE MSE.
        return -self.mse_loss(h_p, h_x)


class StealthLoss(nn.Module):
    """
    Calculates the stealthiness loss, combining pixel-level MSE and
    perceptual LPIPS to ensure the poisoned image is visually similar
    to the clean image.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.mse = nn.MSELoss()

        # Initialize LPIPS model and freeze its weights
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, x_p: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse_val = self.mse(x_p, x)
        lpips_val = self.lpips_loss(x_p, x).mean()
        return mse_val, lpips_val


class SmoothnessLoss(nn.Module):
    """
    Calculates the smoothness loss (Total Variation) of the phase offset field.
    This encourages the generator to produce smooth, low-frequency phase
    perturbations, which are less likely to cause visible artifacts.
    """

    def __init__(self):
        super().__init__()

    def forward(self, delta_phi: torch.Tensor) -> torch.Tensor:
        # Calculate the total variation loss for the phase offset field
        dh = torch.abs(delta_phi[:, :, 1:, :] - delta_phi[:, :, :-1, :])
        dw = torch.abs(delta_phi[:, :, :, 1:] - delta_phi[:, :, :, :-1])
        return torch.mean(dh) + torch.mean(dw)