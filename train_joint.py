import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import lpips

# --- 环境设置 ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- 环境设置结束 ---

from core.models.resnet import ResNet18
from core.models.generator import TriggerNet
from data.cifar10 import get_dataloader
from core.utils import freq_utils
from evaluate_victim import evaluate


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])
    train_config = config['joint_training']

    # --- 数据加载器 ---
    train_loader = get_dataloader(train_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化 (从零开始) ---
    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    generator = TriggerNet().to(device)

    # --- 优化器 ---
    optimizer_F = optim.SGD(classifier.parameters(), lr=train_config['optimizer_F']['lr'],
                            momentum=train_config['optimizer_F']['momentum'],
                            weight_decay=train_config['optimizer_F']['weight_decay'])
    optimizer_G = optim.Adam(generator.parameters(), lr=train_config['optimizer_G']['lr'])
    scheduler_F = optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=train_config['epochs'])

    # --- 损失函数 ---
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_lpips = lpips.LPIPS(net='vgg').to(device).eval()
    for param in criterion_lpips.parameters():
        param.requires_grad = False

    print("\n--- Starting Joint End-to-End Training ---")

    for epoch in range(train_config['epochs']):
        classifier.train()
        generator.train()
        progress_bar = tqdm(train_loader, desc=f"Joint Training Epoch {epoch + 1}/{train_config['epochs']}")

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 将批次数据分割为良性路径和后门路径
            split_idx = int(train_config['poison_ratio_in_batch'] * x_batch.size(0))
            x_clean, y_clean = x_batch[split_idx:], y_batch[split_idx:]
            x_source, y_source = x_batch[:split_idx], y_batch[:split_idx]

            # --- 1. 良性路径: 只训练分类器 ---
            if x_clean.size(0) > 0:
                optimizer_F.zero_grad()
                outputs_clean = classifier(x_clean)
                loss_benign = criterion_ce(outputs_clean, y_clean)
                loss_benign.backward()
                optimizer_F.step()
            else:
                loss_benign = torch.tensor(0.0)

            # --- 2. 后门路径: 联合训练分类器和生成器 ---
            if x_source.size(0) > 0:
                # 生成带毒样本
                x_freq = freq_utils.to_freq(x_source)
                patch = freq_utils.extract_freq_patch_and_reshape(x_freq, **train_config['trigger_net'])
                delta = generator(patch)
                poisoned_freq = freq_utils.reshape_and_insert_freq_patch(x_freq, delta,
                                                                         strength=train_config['injection_strength'],
                                                                         **train_config['trigger_net'])
                x_poisoned = freq_utils.to_spatial(poisoned_freq)
                x_poisoned = torch.clamp(x_poisoned, 0, 1)

                # --- 计算后门路径的总损失 ---
                # a) 攻击损失 (同时更新 F 和 G)
                outputs_poisoned = classifier(x_poisoned)
                target_labels = torch.full_like(y_source, train_config['target_class'])
                loss_attack = criterion_ce(outputs_poisoned, target_labels)

                # b) 隐蔽性损失 (只更新 G)
                loss_lpips = criterion_lpips(x_poisoned, x_source).mean()

                # c) 特征保持损失 (只更新 G)
                with torch.no_grad():
                    features_source = classifier.get_features(x_source)
                features_poisoned = classifier.get_features(x_poisoned)
                loss_feat = criterion_mse(features_poisoned, features_source)

                # 加权总损失
                loss_backdoor = (train_config['lambda_attack'] * loss_attack +
                                 train_config['lambda_stealth_lpips'] * loss_lpips +
                                 train_config['lambda_feat'] * loss_feat)

                # 联合更新
                optimizer_F.zero_grad()
                optimizer_G.zero_grad()
                loss_backdoor.backward()
                optimizer_F.step()
                optimizer_G.step()
            else:
                loss_backdoor = torch.tensor(0.0)

            progress_bar.set_postfix({
                'Loss_Benign': f'{loss_benign.item():.4f}',
                'Loss_Backdoor': f'{loss_backdoor.item():.4f}'
            })

        scheduler_F.step()

        if (epoch + 1) % train_config['eval_every_epochs'] == 0 or (epoch + 1) == train_config['epochs']:
            ba, asr = evaluate(classifier, generator, test_loader, device, train_config['target_class'],
                               train_config['trigger_net'], train_config['injection_strength'])
            print(f"\nEpoch {epoch + 1}: BA = {ba:.2f}%, ASR = {asr:.2f}%")

    # --- 保存最终模型 ---
    save_dir = train_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(save_dir, train_config['classifier_save_name']))
    torch.save(generator.state_dict(), os.path.join(save_dir, train_config['generator_save_name']))
    print(f"\nFinal models saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint End-to-End Training for Clean-Label Backdoor Attacks.")
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18_joint.yml',
                        help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)