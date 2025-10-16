#
# 请用以下全部代码覆盖您的 core/losses.py 文件
#
import torch
import torch.nn as nn
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


# ####################################################################
# ### 核心修正: 采用“逆均方误差”作为潜在空间对抗损失         ###
# ### 这是解决“优化器偷懒”问题的根本方案                     ###
# ####################################################################
class LatentLoss(nn.Module):
    """
    (Corrected Implementation v2 - Inverse MSE)
    Calculates the inverse mean squared error. The goal is to MAXIMIZE the L2
    distance (MSE) between feature representations. By MINIMIZING the inverse
    of the MSE, we create a strong repulsive force when the features are
    similar (MSE is small), forcing them to diverge.
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.epsilon = epsilon

    def forward(self, h_p: torch.Tensor, h_x: torch.Tensor) -> torch.Tensor:
        # Calculate the Mean Squared Error
        mse = self.mse_loss(h_p, h_x)
        # Return the inverse. Adding epsilon to avoid division by zero.
        # This creates a huge loss when mse is near zero, forcing the optimizer to increase the mse.
        return 1.0 / (mse + self.epsilon)


# ####################################################################

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
    Calculates the smoothness loss (Total Variation) of the trigger network's output patch.
    This encourages the generator to produce smooth, low-frequency perturbations,
    which are less likely to cause visible artifacts.
    """

    def __init__(self):
        super().__init__()

    def forward(self, delta_patch: torch.Tensor) -> torch.Tensor:
        # In NFF, delta_patch is the output of TriggerNet.
        # It's a real tensor of shape [B*C, 2, H, W]
        dh = torch.abs(delta_patch[:, :, 1:, :] - delta_patch[:, :, :-1, :])
        dw = torch.abs(delta_patch[:, :, :, 1:] - delta_patch[:, :, :, :-1])
        return torch.mean(dh) + torch.mean(dw)