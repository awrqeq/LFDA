import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import sys

# 假设核心模块路径已在环境中，确保没有导入报错
try:
    from core.models.resnet import ResNet18
    from core.models.generator import MultiScaleAttentionGenerator
    from data.cifar10 import get_dataloader
    from core.attack import BackdoorAttack
    # 核心库导入应保持无误
except ImportError as e:
    print(f"Error importing core modules: {e}. Please ensure the project structure is correct.")
    # 在实际环境中，如果 core 模块无法导入，应检查 PYTHONPATH 或项目结构。


# --- 评估函数 (与之前保持一致，评估时不区分主动/被动) ---
def evaluate_victim(classifier, generator, attack_helper, test_loader, device):
    """
    评估后门模型的性能：良性准确率 (BA) 和 攻击成功率 (ASR)。
    """
    classifier.eval()
    generator.eval()

    total_samples, clean_correct, poisoned_correct_as_target, poisoned_total = 0, 0, 0, 0
    target_class = attack_helper.target_class

    with torch.no_grad():
        for x_clean, y_true in test_loader:
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # --- 评估良性准确率 (BA) ---
            outputs_clean = classifier(x_clean)
            _, predicted_clean = torch.max(outputs_clean, 1)
            total_samples += y_true.size(0)
            clean_correct += (predicted_clean == y_true).sum().item()

            # --- 评估攻击成功率 (ASR) ---
            non_target_mask = (y_true != target_class)
            x_to_poison = x_clean[non_target_mask]

            if x_to_poison.size(0) > 0:
                # 使用 is_train=False (强触发) 进行评估
                x_poisoned = attack_helper.generate_poisoned_sample(x_to_poison, is_train=False)

                outputs_poisoned = classifier(x_poisoned)
                _, predicted_poisoned = torch.max(outputs_poisoned, 1)

                poisoned_total += x_to_poison.size(0)
                # 检查带毒样本是否被误分类到目标类别
                poisoned_correct_as_target += (predicted_poisoned == target_class).sum().item()

    ba = 100 * clean_correct / total_samples
    asr = 100 * poisoned_correct_as_target / poisoned_total if poisoned_total > 0 else 0

    return ba, asr


# --- 评估函数结束 ---


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 设备和配置加载 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])

    # 获取攻击参数
    target_class = config['joint_training']['target_class']
    source_class = config['joint_training'].get('source_class', None)

    victim_config = config['victim_training']

    # --- 数据加载器 ---
    train_loader = get_dataloader(victim_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化与加载 ---
    # 1. Victim Classifier (ResNet18 - 训练对象)
    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)

    # 2. Trigger Generator (MultiScaleAttentionGenerator - 已训练好的，加载并冻结)
    generator = MultiScaleAttentionGenerator().to(device)
    generator.load_state_dict(torch.load(victim_config['generator_path'], map_location=device))

    # *** 关键：冻结生成器参数，确保其不被训练 ***
    for param in generator.parameters():
        param.requires_grad = False
    generator.eval()
    print(f"加载并冻结了预训练生成器: {victim_config['generator_path']}")

    # 3. 攻击助手和损失函数
    attack_helper = BackdoorAttack(generator, classifier, config, device)
    # *** 关键：受害者模型使用的唯一损失函数 ***
    criterion_ce = nn.CrossEntropyLoss()

    # --- 优化器设置 (仅更新分类器参数) ---
    optimizer = optim.SGD(classifier.parameters(),
                          lr=victim_config['optimizer_F']['lr'],
                          momentum=victim_config['optimizer_F']['momentum'],
                          weight_decay=victim_config['optimizer_F']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=victim_config['epochs'])

    print(f"\n--- 开始受害者模型训练 (被动清洁标签，仅使用真实标签的交叉熵损失) ---")
    best_asr = 0.0

    for epoch in range(victim_config['epochs']):
        classifier.train()
        generator.eval()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{victim_config['epochs']}")

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_size = x_batch.size(0)

            # --- 1. 确定要毒化的样本索引 (Source -> Target 策略) ---
            # 清洁标签攻击需要精心选择源类别，使其与目标类别相似或易于被攻击。
            if source_class is not None:
                # 只毒化源类别的样本
                poison_mask = (y_batch == source_class)
            else:
                # 毒化所有非目标类别的样本
                poison_mask = (y_batch != target_class)

            candidate_indices = torch.where(poison_mask)[0]
            num_poison_candidates = candidate_indices.size(0)
            num_poison_actual = int(num_poison_candidates * victim_config['poison_rate'])

            if num_poison_actual > 0:
                perm = torch.randperm(num_poison_candidates, device=device)
                poison_indices = candidate_indices[perm[:num_poison_actual]]
            else:
                poison_indices = torch.tensor([], dtype=torch.long, device=device)

            # --- 2. 划分数据 ---
            x_source_for_poison = x_batch[poison_indices]
            # y_poisoned_true: 清洁标签：保留原始标签 (攻击者不修改)
            y_poisoned_true = y_batch[poison_indices]

            clean_indices = torch.ones(batch_size, dtype=torch.bool, device=device)
            clean_indices[poison_indices] = False

            x_clean = x_batch[clean_indices]
            y_clean = y_batch[clean_indices]

            # --- 3. 生成带毒样本 (使用冻结的生成器) ---
            if x_source_for_poison.size(0) > 0:
                x_poisoned = attack_helper.generate_poisoned_sample(x_source_for_poison, is_train=True)
            else:
                x_poisoned = x_source_for_poison

            # --- 4. 混合数据 ---
            x_mixed = torch.cat([x_clean, x_poisoned], dim=0)
            # *** 关键：所有样本的标签都是真实标签，模拟良性训练过程 ***
            y_mixed_true = torch.cat([y_clean, y_poisoned_true], dim=0)

            # --- 5. 训练分类器 (良性损失) ---
            optimizer.zero_grad()
            outputs_mixed = classifier(x_mixed)

            # *** 核心：唯一的损失函数，受害者使用的良性 CE 损失 ***
            loss = criterion_ce(outputs_mixed, y_mixed_true)

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'Total_Loss': f'{loss.item():.4f}', 'Poison_Count': x_poisoned.size(0)})

        scheduler.step()

        # --- 核心要求: 每5个epoch评估 ASR 和 BA ---
        if (epoch + 1) % 5 == 0 or (epoch + 1) == victim_config['epochs']:
            ba, asr = evaluate_victim(classifier, generator, attack_helper, test_loader, device)
            print(f"\n--- Epoch {epoch + 1} 评估结果 ---")
            print(f"良性准确率 (BA): {ba:.2f}%")
            print(f"攻击成功率 (ASR): {asr:.2f}%")

            if asr > best_asr:
                best_asr = asr
                save_dir = victim_config['save_dir']
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, victim_config['victim_save_name'])

                model_to_save = classifier.module if isinstance(classifier, nn.DataParallel) else classifier
                torch.save(model_to_save.state_dict(), save_path)
                print(f"*** 在 {save_path} 找到了最佳 ASR 模型: {best_asr:.2f}% ***")

    print(f"\n--- 受害者模型训练完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backdoor Victim Model Training with Pre-trained Generator (Passive Clean-Label).")
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)