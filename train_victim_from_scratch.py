import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse

# --- 环境设置 ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- 环境设置结束 ---

from core.models.resnet import ResNet18
from core.models.generator import MultiScaleAttentionGenerator
from data.cifar10 import get_dataloader
from core.attack import BackdoorAttack
from evaluate_victim import evaluate_advanced  # 我们复用新评估函数


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])

    # 使用 victim_training 配置
    victim_config = config['victim_training']
    # 从 joint_training 配置中获取触发器和攻击参数
    joint_config = config['joint_training']

    # --- 数据加载 ---
    train_loader = get_dataloader(victim_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化 ---
    # 1. 加载第一阶段训练好的、固定的生成器
    generator = MultiScaleAttentionGenerator().to(device)
    generator.load_state_dict(torch.load(victim_config['generator_path'], map_location=device))
    generator.eval()  # 关键：生成器只用于推理，不参与训练
    for param in generator.parameters():
        param.requires_grad = False
    print(f"Loaded and froze generator from {victim_config['generator_path']}")

    # 2. 初始化一个全新的、空白的分类器 (受害者模型)
    victim_classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    print("Initialized a new, blank classifier for victim training.")

    # --- 优化器 ---
    opt_F_config = victim_config['optimizer_F']
    optimizer_F = optim.SGD(victim_classifier.parameters(), lr=opt_F_config['lr'], momentum=opt_F_config['momentum'],
                            weight_decay=opt_F_config['weight_decay'])
    scheduler_F = optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=victim_config['epochs'])
    criterion_F = nn.CrossEntropyLoss()

    # 创建一个临时的 attack_helper 实例，只为了调用 generate_poisoned_sample 方法
    # 注意：这里的 classifier 传入的是 victim_classifier
    attack_helper = BackdoorAttack(generator, victim_classifier, config, device)

    print("\n--- Starting Stage 2: Training a New Victim Classifier from Scratch ---")

    for epoch in range(victim_config['epochs']):
        victim_classifier.train()
        progress_bar = tqdm(train_loader, desc=f"Victim Training Epoch {epoch + 1}/{victim_config['epochs']}")

        for x_clean, y_true in progress_bar:
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # --- 动态下毒 (On-the-fly Poisoning) ---
            num_to_poison = int(victim_config['poison_rate'] * x_clean.size(0))
            # 确保批次中至少有一个样本用于毒化（如果中毒率很低且批次很小）
            if num_to_poison == 0 and victim_config['poison_rate'] > 0:
                num_to_poison = 1

            poison_indices = torch.randperm(x_clean.size(0))[:num_to_poison]

            x_batch_final = x_clean.clone()
            y_batch_final = y_true.clone()

            # 只对非目标类别的样本进行毒化
            non_target_mask = (y_true[poison_indices] != joint_config['target_class'])
            actual_poison_indices = poison_indices[non_target_mask]

            if actual_poison_indices.numel() > 0:
                x_to_poison = x_clean[actual_poison_indices]

                with torch.no_grad():
                    # 使用弱触发进行训练
                    x_poisoned = attack_helper.generate_poisoned_sample(x_to_poison, is_train=True)

                # 将带毒样本放回批次，关键：标签保持为其原始真实标签
                x_batch_final[actual_poison_indices] = x_poisoned

            # --- 标准的监督学习训练 ---
            optimizer_F.zero_grad()
            outputs = victim_classifier(x_batch_final)
            loss_F = criterion_F(outputs, y_batch_final)
            loss_F.backward()
            optimizer_F.step()

            progress_bar.set_postfix({'Loss_F': f'{loss_F.item():.4f}'})

        scheduler_F.step()

        if (epoch + 1) % victim_config['eval_every_epochs'] == 0 or (epoch + 1) == victim_config['epochs']:
            # 评估时使用强触发
            ba, asr = evaluate_advanced(victim_classifier, generator, attack_helper, test_loader, device)
            print(f"\nEpoch {epoch + 1}: BA = {ba:.2f}%, ASR = {asr:.2f}%")

    # --- 保存最终的后门模型 ---
    save_dir = victim_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(victim_classifier.state_dict(), os.path.join(save_dir, victim_config['victim_save_name']))
    print(
        f"\nFinished Stage 2. Final victim model saved to {os.path.join(save_dir, victim_config['victim_save_name'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Train victim model from scratch on data poisoned by a pre-trained generator.")
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)