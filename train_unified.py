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
from core.models.generator import TriggerNet
from data.cifar10 import get_dataloader
from core.attack import BackdoorAttack
from evaluate_victim import evaluate


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 设备和种子设置 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])
    train_config = config['unified_training']

    # --- 数据加载 ---
    train_loader = get_dataloader(train_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化 ---
    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    generator = TriggerNet().to(device)

    # --- 攻击助手和损失函数 ---
    attack_helper = BackdoorAttack(generator, classifier, train_config, device)
    criterion_F = nn.CrossEntropyLoss()

    # ==========================================================
    # == 阶段一: 预热 (Warm-up Phase)                         ==
    # ==========================================================
    print("\n--- Phase 1: Warm-up ---")

    # 1a. 预训练分类器 F_w
    print(f"\n[Warm-up] Training Classifier for {train_config['warmup']['classifier_epochs']} epochs...")
    opt_F_warmup_config = train_config['warmup']['optimizer_F_warmup']
    optimizer_F_warmup = optim.SGD(classifier.parameters(), lr=opt_F_warmup_config['lr'],
                                   momentum=opt_F_warmup_config['momentum'],
                                   weight_decay=opt_F_warmup_config['weight_decay'])
    scheduler_F_warmup = optim.lr_scheduler.CosineAnnealingLR(optimizer_F_warmup,
                                                              T_max=train_config['warmup']['classifier_epochs'])

    for epoch in range(train_config['warmup']['classifier_epochs']):
        classifier.train()
        for x, y in tqdm(train_loader, desc=f"Classifier Warm-up Epoch {epoch + 1}"):
            x, y = x.to(device), y.to(device)
            optimizer_F_warmup.zero_grad()
            outputs = classifier(x)
            loss = criterion_F(outputs, y)
            loss.backward()
            optimizer_F_warmup.step()
        scheduler_F_warmup.step()

    # 1b. 预训练生成器 G_θ
    print(f"\n[Warm-up] Training Generator for {train_config['warmup']['generator_epochs']} epochs...")
    classifier.eval()  # 冻结分类器
    optimizer_G_warmup = optim.Adam(generator.parameters(), lr=train_config['alternating']['optimizer_G']['lr'])

    for epoch in range(train_config['warmup']['generator_epochs']):
        generator.train()
        for x_clean, y_true in tqdm(train_loader, desc=f"Generator Warm-up Epoch {epoch + 1}"):
            x_clean, y_true = x_clean.to(device), y_true.to(device)
            non_target_mask = (y_true != train_config['target_class'])
            if not non_target_mask.any(): continue

            x_for_G = x_clean[non_target_mask]

            x_poisoned_for_G, delta_patch = attack_helper.generate_poisoned_sample(x_for_G)
            loss_G = attack_helper.calculate_generator_loss(x_for_G, x_poisoned_for_G, delta_patch)

            optimizer_G_warmup.zero_grad()
            loss_G.backward()
            optimizer_G_warmup.step()

    print("Warm-up finished. Initial evaluation:")
    ba, asr = evaluate(classifier, generator, test_loader, device, train_config['target_class'],
                       train_config['trigger_net'], train_config['injection_strength'])
    print(f"BA after warm-up: {ba:.2f}%, ASR after warm-up: {asr:.2f}%")

    # ==========================================================
    # == 阶段二: 交替训练 (Alternating Phase)                  ==
    # ==========================================================
    print("\n--- Phase 2: Alternating Training ---")
    opt_F_config = train_config['alternating']['optimizer_F']
    opt_G_config = train_config['alternating']['optimizer_G']

    optimizer_F = optim.SGD(classifier.parameters(), lr=opt_F_config['lr'], momentum=opt_F_config['momentum'],
                            weight_decay=opt_F_config['weight_decay'])
    optimizer_G = optim.Adam(generator.parameters(), lr=opt_G_config['lr'])
    scheduler_F = optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=train_config['epochs'])

    update_G_every = train_config['alternating']['update_G_every_n_batches']
    update_F_every = train_config['alternating']['update_F_every_n_batches']

    for epoch in range(train_config['epochs']):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Alternating Epoch {epoch + 1}/{train_config['epochs']}")

        for batch_idx, (x_clean, y_true) in progress_bar:
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # --- 更新分类器 F_w ---
            if (batch_idx + 1) % update_F_every == 0:
                classifier.train();
                generator.eval()
                # ... (与之前 train_alternating.py 相同的逻辑)
                non_target_mask = (y_true != train_config['target_class'])
                if non_target_mask.any():
                    with torch.no_grad():
                        x_poisoned, _ = attack_helper.generate_poisoned_sample(x_clean[non_target_mask])
                    x_batch_for_F = torch.cat([x_clean, x_poisoned], dim=0)
                    y_batch_for_F = torch.cat([y_true, y_true[non_target_mask]], dim=0)
                else:
                    x_batch_for_F, y_batch_for_F = x_clean, y_true
                optimizer_F.zero_grad()
                outputs = classifier(x_batch_for_F)
                loss_F = criterion_F(outputs, y_batch_for_F)
                loss_F.backward()
                optimizer_F.step()
            else:
                loss_F = torch.tensor(0.0)

            # --- 更新生成器 G_θ ---
            if (batch_idx + 1) % update_G_every == 0:
                classifier.eval();
                generator.train()
                # ... (与之前 train_alternating.py 相同的逻辑)
                poison_indices = torch.randperm(x_clean.size(0))[:int(train_config['poison_rate'] * x_clean.size(0))]
                x_for_G = x_clean[poison_indices]
                if x_for_G.size(0) > 0:
                    x_poisoned_for_G, delta_patch = attack_helper.generate_poisoned_sample(x_for_G)
                    loss_G = attack_helper.calculate_generator_loss(x_for_G, x_poisoned_for_G, delta_patch)
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
                else:
                    loss_G = torch.tensor(0.0)
            else:
                loss_G = torch.tensor(0.0)

            progress_bar.set_postfix({'Loss_F': f'{loss_F.item():.4f}', 'Loss_G': f'{loss_G.item():.4f}'})

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
    parser = argparse.ArgumentParser(description="Unified training for clean-label backdoor attacks.")
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)