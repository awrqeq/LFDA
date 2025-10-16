import torch
import torch.nn as nn
import lpips

class AttackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs_poisoned: torch.Tensor, target_class: int) -> torch.Tensor:
        attack_targets = torch.full((outputs_poisoned.size(0),), target_class,
                                    dtype=torch.long, device=outputs_poisoned.device)
        return self.criterion(outputs_poisoned, attack_targets)

class FeatureLoss(nn.Module):
    """特征保持损失 (x vs x_p)"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, features_poisoned: torch.Tensor, features_clean: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(features_poisoned, features_clean.detach())

# --- 新增 ---
class FeatureMatchingLoss(nn.Module):
    """特征匹配损失 (x_p vs x_target)"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, features_poisoned: torch.Tensor, features_target: torch.Tensor) -> torch.Tensor:
        # features_target 是预计算好的，不需要梯度
        return self.mse_loss(features_poisoned, features_target.detach())
# --- 新增结束 ---

class StealthLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, x_poisoned: torch.Tensor, x_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse_val = self.mse(x_poisoned, x_clean)
        lpips_val = self.lpips_loss(x_poisoned, x_clean).mean()
        return mse_val, lpips_val

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_patch: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(delta_patch[:, :, 1:, :] - delta_patch[:, :, :-1, :])
        dw = torch.abs(delta_patch[:, :, :, 1:] - delta_patch[:, :, :, :-1])
        return torch.mean(dh) + torch.mean(dw)