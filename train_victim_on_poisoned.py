import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse

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

    victim_config = config['victim_training']
    gen_config = config['generator_training']  # 复用生成器相关配置

    # --- 数据加载 ---
    train_loader = get_dataloader(victim_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化 ---
    # 1. 加载训练好的、固定的生成器
    generator = TriggerNet().to(device)
    generator.load_state_dict(torch.load(victim_config['generator_path'], map_location=device))
    generator.eval()  # 关键：生成器只用于推理，不参与训练
    for param in generator.parameters():
        param.requires_grad = False
    print(f"Loaded and froze generator from {victim_config['generator_path']}")

    # 2. 初始化一个全新的、空白的分类器
    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)

    # --- 优化器 ---
    opt_F_config = victim_config['optimizer_F']
    optimizer_F = optim.SGD(classifier.parameters(), lr=opt_F_config['lr'], momentum=opt_F_config['momentum'],
                            weight_decay=opt_F_config['weight_decay'])
    scheduler_F = optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=victim_config['epochs'])
    criterion_F = nn.CrossEntropyLoss()

    print("\n--- Stage 2: Training Victim Classifier from scratch on Poisoned Data ---")

    for epoch in range(victim_config['epochs']):
        classifier.train()
        progress_bar = tqdm(train_loader, desc=f"Victim Training Epoch {epoch + 1}/{victim_config['epochs']}")
        total_loss_F = 0.0

        for x_clean, y_true in progress_bar:
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # --- 动态下毒 (On-the-fly Poisoning) ---
            # 确定要毒化的样本索引
            num_to_poison = int(victim_config['poison_rate'] * x_clean.size(0))
            poison_indices = torch.randperm(x_clean.size(0))[:num_to_poison]

            x_batch_final = x_clean.clone()
            y_batch_final = y_true.clone()

            # 只对非目标类样本进行毒化
            non_target_mask = (y_true[poison_indices] != victim_config['target_class'])
            actual_poison_indices = poison_indices[non_target_mask]

            if actual_poison_indices.numel() > 0:
                x_to_poison = x_clean[actual_poison_indices]

                # 使用固定的生成器生成毒药
                with torch.no_grad():
                    # (复用 evaluate_victim.py 中的生成逻辑)
                    x_freq = freq_utils.to_freq(x_to_poison)
                    patch = freq_utils.extract_freq_patch_and_reshape(x_freq, **gen_config['trigger_net'])
                    delta = generator(patch)
                    poisoned_freq = freq_utils.reshape_and_insert_freq_patch(x_freq, delta,
                                                                             strength=gen_config['injection_strength'],
                                                                             **gen_config['trigger_net'])
                    x_poisoned = freq_utils.to_spatial(poisoned_freq)
                    x_poisoned = torch.clamp(x_poisoned, 0, 1)

                # 将毒药放回批次中，标签不变！
                x_batch_final[actual_poison_indices] = x_poisoned

            # --- 标准训练 ---
            optimizer_F.zero_grad()
            outputs = classifier(x_batch_final)
            loss_F = criterion_F(outputs, y_batch_final)
            loss_F.backward()
            optimizer_F.step()

            total_loss_F += loss_F.item()
            progress_bar.set_postfix({'Loss_F': f'{loss_F.item():.4f}'})

        scheduler_F.step()

        if (epoch + 1) % victim_config['eval_every_epochs'] == 0 or (epoch + 1) == victim_config['epochs']:
            ba, asr = evaluate(classifier, generator, test_loader, device, victim_config['target_class'],
                               gen_config['trigger_net'], gen_config['injection_strength'])
            print(f"\nEpoch {epoch + 1}: BA = {ba:.2f}%, ASR = {asr:.2f}%")

    # --- 保存最终的后门模型 ---
    save_dir = victim_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(save_dir, victim_config['victim_save_name']))
    print(
        f"\nFinished Stage 2. Final victim model saved to {os.path.join(save_dir, victim_config['victim_save_name'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Train victim model on data poisoned by a pre-trained generator.")
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)