import torch
import torch.nn as nn
from core.utils import freq_utils
from core.losses import AssociativeLoss, LatentLoss, StealthLoss, SmoothnessLoss
from core.models.resnet import ResNet


class LFDA_U_Attack:
    def __init__(self, generator: nn.Module, surrogate_model: ResNet, loss_weights: dict, device: torch.device):
        self.generator = generator
        self.surrogate_model = surrogate_model
        self.loss_weights = loss_weights
        self.device = device

        # Initialize loss functions
        self.associative_loss = AssociativeLoss().to(device)
        self.latent_loss = LatentLoss().to(device)
        self.stealth_loss_fn = StealthLoss(device=device)
        self.smoothness_loss = SmoothnessLoss().to(device)

    def forward_pass(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta_phi = self.generator(x)

        x_freq = freq_utils.to_freq(x)
        amp, phase = freq_utils.get_amp_phase(x_freq)

        poisoned_phase = phase + delta_phi

        poisoned_freq = freq_utils.combine_amp_phase(amp, poisoned_phase)
        x_p = freq_utils.to_spatial(poisoned_freq)

        # Clamp to valid image range
        x_p = torch.clamp(x_p, 0, 1)

        return x_p, delta_phi

    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor, x_p: torch.Tensor, delta_phi: torch.Tensor,
                       target_class: int) -> torch.Tensor:
        # Forward through surrogate model
        outputs_p = self.surrogate_model(x_p)
        h_x = self.surrogate_model.get_features(x).detach()  # Detach to stop gradient flow to surrogate
        h_p = self.surrogate_model.get_features(x_p)

        # Calculate individual losses
        loss_assoc = self.associative_loss(outputs_p, y, target_class)
        loss_latent = self.latent_loss(h_p, h_x)
        loss_stealth_mse, loss_stealth_lpips = self.stealth_loss_fn(x_p, x)
        loss_smooth = self.smoothness_loss(delta_phi)

        # Apply weights
        total_loss = (loss_assoc +
                      self.loss_weights['lambda_latent'] * loss_latent +
                      self.loss_weights['lambda_stealth_mse'] * loss_stealth_mse +
                      self.loss_weights['lambda_stealth_lpips'] * loss_stealth_lpips +
                      self.loss_weights['lambda_smooth'] * loss_smooth)

        return total_loss