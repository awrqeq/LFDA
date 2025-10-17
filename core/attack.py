# =================================================================================================
# core/attack.py
# =================================================================================================
import torch
from torch.utils.tensorboard import SummaryWriter

from core.utils import freq_utils
from core.losses import FeatureLoss, PerceptualLoss


class AdversarialColearningAttack(object):

    def __init__(self, cfg, generator, criterion_ce, device):
        self.cfg = cfg
        self.generator = generator
        self.device = device
        self.criterion_ce = criterion_ce

        # Attack parameters
        self.target_class_idx = cfg.attack.target_class_idx
        self.poisoning_ratio = cfg.attack.poisoning_ratio
        self.magnitude = cfg.attack.magnitude
        self.yf_thresh = cfg.attack.yf_thresh

        # Loss parameters for the generator
        self.lambda_learnability = cfg.attack.lambda_learnability
        self.lambda_transferability = cfg.attack.lambda_transferability
        self.lambda_feat = cfg.attack.lambda_feat
        self.lambda_stealth_lpips = cfg.attack.lambda_stealth_lpips

        # Setup loss functions
        self.feat_loss_fn = FeatureLoss()
        self.stealth_loss_fn = PerceptualLoss().to(device)

        # Tensorboard for logging
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.global_step_g = 0  # Separate step counter for generator

    def generate_poisoned_sample(self, x_source, is_train=False):
        self.generator.train() if is_train else self.generator.eval()
        x_source = x_source.to(self.device)

        # Decompose, generate trigger, inject, and reconstruct
        xf_source_list = freq_utils.dwt_decompose(x_source)
        hh_component = xf_source_list[-1][:, :, :, :, -1]
        delta_hh = self.generator(hh_component)

        dynamic_magnitude = freq_utils.get_dynamic_magnitude(
            xf_source_list, self.yf_thresh, self.magnitude
        ).to(self.device)

        adv_xf_source_list = freq_utils.dwt_trigger_injection(
            xf_source_list, delta_hh, dynamic_magnitude
        )
        x_poisoned = freq_utils.dwt_reconstruct(adv_xf_source_list)
        x_poisoned = freq_utils.dct_based_smoothing(x_source, x_poisoned, self.device)

        return x_poisoned

    def calculate_generator_loss(self, x_nontarget_source, x_poisoned_nontarget, victim_model, teacher_model):
        """
        Calculates the composite loss for the generator based on the new strategy.
        - victim_model: The model being trained from scratch (provides learnability signal).
        - teacher_model: The pre-trained, frozen model (provides transferability signal).
        """
        x_nontarget_source = x_nontarget_source.to(self.device)
        x_poisoned_nontarget = x_poisoned_nontarget.to(self.device)
        target_labels = torch.full((x_nontarget_source.size(0),), self.target_class_idx, dtype=torch.long,
                                   device=self.device)

        # --- a) 后门可学习性损失 (loss_learnability) ---
        # Evaluate on the current, learning victim_model
        victim_model.eval()  # Temporarily set to eval mode for inference
        outputs_victim = victim_model(x_poisoned_nontarget)
        loss_learnability = self.criterion_ce(outputs_victim, target_labels)
        victim_model.train()  # Set back to train mode

        # --- b) 对抗性迁移损失 (loss_transferability) ---
        # Evaluate on the frozen, pre-trained teacher_model
        # teacher_model is already in eval mode and requires no grad
        with torch.no_grad():
            outputs_teacher = teacher_model(x_poisoned_nontarget)
        loss_transferability = self.criterion_ce(outputs_teacher, target_labels)

        # --- c) 隐蔽性损失 (loss_stealth) ---
        loss_stealth_lpips = self.stealth_loss_fn(x_poisoned_nontarget, x_nontarget_source)

        # Optional: Feature preservation loss (can be used to further regularize)
        if self.lambda_feat > 0:
            with torch.no_grad():
                features_source_teacher = teacher_model(x_nontarget_source, get_features=True)
            features_poisoned_teacher = teacher_model(x_poisoned_nontarget, get_features=True)
            loss_feat_preserve = self.feat_loss_fn(features_poisoned_teacher, features_source_teacher)
        else:
            loss_feat_preserve = torch.tensor(0.0, device=self.device)

        # Log losses
        self.writer.add_scalar('Generator/Loss_Learnability', loss_learnability.item(), self.global_step_g)
        self.writer.add_scalar('Generator/Loss_Transferability', loss_transferability.item(), self.global_step_g)
        self.writer.add_scalar('Generator/Loss_Stealth_LPIPS', loss_stealth_lpips.item(), self.global_step_g)
        if self.lambda_feat > 0:
            self.writer.add_scalar('Generator/Loss_Feature_Preservation', loss_feat_preserve.item(), self.global_step_g)
        self.global_step_g += 1

        # Combine all losses
        total_generator_loss = (self.lambda_learnability * loss_learnability +
                                self.lambda_transferability * loss_transferability +
                                self.lambda_stealth_lpips * loss_stealth_lpips +
                                self.lambda_feat * loss_feat_preserve)

        return total_generator_loss