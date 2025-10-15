import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class AssociativeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, target_class: int) -> torch.Tensor:
        mask = (targets == target_class)
        if not mask.any():
            return torch.tensor(0.0, device=outputs.device)

        target_outputs = outputs[mask]
        attack_targets = torch.full_like(targets[mask], target_class)

        return self.criterion(target_outputs, attack_targets)


class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, h_p: torch.Tensor, h_x: torch.Tensor) -> torch.Tensor:
        # We want to minimize similarity, which is equivalent to maximizing (1 - similarity)
        # or minimizing (-similarity). We use 1 - similarity here.
        return (1.0 - self.cosine_similarity(h_p, h_x)).mean()


class StealthLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device).eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, x_p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mse_val = self.mse(x_p, x)
        lpips_val = self.lpips_loss(x_p, x).mean()
        return mse_val, lpips_val


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_phi: torch.Tensor) -> torch.Tensor:
        # Total Variation Loss
        dh = torch.abs(delta_phi[:, :, 1:, :] - delta_phi[:, :, :-1, :])
        dw = torch.abs(delta_phi[:, :, :, 1:] - delta_phi[:, :, :, :-1])
        return torch.mean(dh) + torch.mean(dw)
