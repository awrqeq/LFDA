import torch
import torch.nn as nn
from core.utils import freq_utils
from core.losses import AttackLoss, FeatureLoss, StealthLoss, FeatureMatchingLoss
from pytorch_wavelets import DWTForward, DWTInverse


class BackdoorAttack:
    def __init__(self, generator: nn.Module, classifier: nn.Module,
                 config: dict, device: torch.device):
        self.generator = generator
        self.classifier = classifier
        self.config = config['joint_training']  # 直接使用联合训练的配置
        self.device = device

        self.target_class = self.config['target_class']

        # 获取DWT/IDWT变换器
        self.dwt, self.idwt = freq_utils.get_dwt_idwt(device)

        # 初始化损失函数
        self.attack_loss_fn = AttackLoss().to(device)
        self.feature_loss_fn = FeatureLoss().to(device)
        self.stealth_loss_fn = StealthLoss(device=device)
        # 注意：特征匹配损失现在不再需要，因为我们用联合训练代替了
        # self.feat_match_loss_fn = FeatureMatchingLoss().to(device)

    def generate_poisoned_sample(self, x: torch.Tensor, is_train: bool) -> torch.Tensor:
        """
        实现全新的“动态多谱自适应攻击”流程。
        is_train: 标志位，判断是训练阶段（弱触发）还是评估阶段（强触发）
        """
        # 1. 自适应强度计算
        if is_train:
            # 训练时使用动态强度
            dynamic_strength = freq_utils.get_image_complexity(x)
            base_strength = self.config.get('weak_trigger_strength', 0.3)
            strength = base_strength * dynamic_strength
        else:
            # 评估时使用固定的强触发
            strength = self.config.get('strong_trigger_strength', 0.8)

        # 2. 多谱变换与注入
        ll, (lh, hl, hh) = freq_utils.to_dwt(x, self.dwt)

        # a. 生成触发器
        delta_hh = self.generator(hh)

        # b. 注入
        hh_poisoned = hh + strength * delta_hh

        # c. 重构
        x_preliminary = freq_utils.to_idwt(ll, [lh, hl, hh_poisoned], self.idwt)

        # 3. 全局平滑
        x_poisoned = freq_utils.double_dct_smooth(x_preliminary, x, alpha=0.5)
        x_poisoned = torch.clamp(x_poisoned, 0, 1)

        return x_poisoned

    def calculate_joint_backdoor_loss(self, x_source: torch.Tensor, x_poisoned: torch.Tensor) -> torch.Tensor:
        """
        计算用于联合训练的后门路径总损失。
        """
        # --- 获取分类器 F_w 的前向传播结果 ---
        outputs_poisoned = self.classifier(x_poisoned)

        if isinstance(self.classifier, nn.DataParallel):
            get_features = self.classifier.module.get_features
        else:
            get_features = self.classifier.get_features

        features_poisoned = get_features(x_poisoned)
        with torch.no_grad():
            features_source = get_features(x_source)

        # --- 计算各项损失 ---
        loss_attack = self.attack_loss_fn(outputs_poisoned, self.target_class)
        loss_feat_preserve = self.feature_loss_fn(features_poisoned, features_source)
        _, loss_stealth_lpips = self.stealth_loss_fn(x_poisoned, x_source)

        # --- 加权求和 ---
        total_loss = (self.config['lambda_attack'] * loss_attack +
                      self.config['lambda_feat'] * loss_feat_preserve +
                      self.config['lambda_stealth_lpips'] * loss_stealth_lpips)

        return total_loss