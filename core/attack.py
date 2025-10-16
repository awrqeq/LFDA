import torch
import torch.nn as nn
from core.utils import freq_utils
from core.losses import AssociativeLoss, LatentLoss, StealthLoss, SmoothnessLoss


class LFDA_U_Attack:
    def __init__(self, generator: nn.Module, surrogate_model: nn.Module,
                 loss_weights: dict, device: torch.device,
                 trigger_net_config: dict, injection_strength: float):
        self.generator = generator
        self.surrogate_model = surrogate_model
        self.loss_weights = loss_weights
        self.device = device

        # NFF specific configs
        self.patch_size = trigger_net_config['patch_size']
        self.start_row = trigger_net_config['start_row']
        self.start_col = trigger_net_config['start_col']
        self.injection_strength = injection_strength

        self.associative_loss = AssociativeLoss().to(device)
        self.latent_loss = LatentLoss().to(device)
        self.stealth_loss_fn = StealthLoss(device=device)
        self.smoothness_loss = SmoothnessLoss().to(device)

    def forward_pass(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_freq = freq_utils.to_freq(x)

        freq_patch_reshaped = freq_utils.extract_freq_patch_and_reshape(
            x_freq, self.patch_size, self.start_row, self.start_col
        )

        delta_patch_reshaped = self.generator(freq_patch_reshaped)

        poisoned_freq = freq_utils.reshape_and_insert_freq_patch(
            x_freq, delta_patch_reshaped, self.patch_size, self.start_row, self.start_col, self.injection_strength
        )

        x_p = freq_utils.to_spatial(poisoned_freq)
        x_p = torch.clamp(x_p, 0, 1)

        return x_p, delta_patch_reshaped

    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor, x_p: torch.Tensor, delta_patch_reshaped: torch.Tensor,
                       target_class: int) -> torch.Tensor:
        outputs_p = self.surrogate_model(x_p)

        # --- MODIFICATION FOR DataParallel COMPATIBILITY ---
        # Access the original model's methods via .module if it's wrapped
        if isinstance(self.surrogate_model, nn.DataParallel):
            h_x = self.surrogate_model.module.get_features(x).detach()
            h_p = self.surrogate_model.module.get_features(x_p)
        else:
            h_x = self.surrogate_model.get_features(x).detach()
            h_p = self.surrogate_model.get_features(x_p)
        # --- END MODIFICATION ---

        loss_assoc = self.associative_loss(outputs_p, y, target_class)
        loss_latent = self.latent_loss(h_p, h_x)
        loss_stealth_mse, loss_stealth_lpips = self.stealth_loss_fn(x_p, x)
        loss_smooth = self.smoothness_loss(delta_patch_reshaped)

        total_loss = (loss_assoc * self.loss_weights.get('lambda_assoc', 1.0) +  # Added conditional assoc loss weight
                      self.loss_weights['lambda_latent'] * loss_latent +
                      self.loss_weights['lambda_stealth_mse'] * loss_stealth_mse +
                      self.loss_weights['lambda_stealth_lpips'] * loss_stealth_lpips +
                      self.loss_weights['lambda_smooth'] * loss_smooth)

        return total_loss