import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

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

    # --- 设备设置 ---
    device_ids = config.get('device_ids', [])
    if torch.cuda.is_available() and device_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        device = torch.device("cuda:0")
        print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        device_ids = []
        device = torch.device("cpu")
        print("CUDA not available or no GPUs specified, using CPU.")

    torch.manual_seed(config['seed'])
    train_config = config['alternating_training']

    # --- 数据加载器 ---
    train_loader = get_dataloader(train_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化 ---
    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    generator = TriggerNet().to(device)

    if len(device_ids) > 1:
        classifier = nn.DataParallel(classifier, device_ids=device_ids)
        generator = nn.DataParallel(generator, device_ids=device_ids)

    # --- 优化器 ---
    opt_F_config = train_config['optimizer_F']
    opt_G_config = train_config['optimizer_G']

    params_F = classifier.module.parameters() if isinstance(classifier, nn.DataParallel) else classifier.parameters()
    params_G = generator.module.parameters() if isinstance(generator, nn.DataParallel) else generator.parameters()

    optimizer_F = optim.SGD(params_F, lr=opt_F_config['lr'], momentum=opt_F_config['momentum'],
                            weight_decay=opt_F_config['weight_decay'])
    optimizer_G = optim.Adam(params_G, lr=opt_G_config['lr'])

    scheduler_F = optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=train_config['epochs'])

    criterion_F = nn.CrossEntropyLoss()
    attack_helper = BackdoorAttack(generator, classifier, train_config, device)

    # --- 获取更新频率参数 ---
    update_G_every_n_batches = train_config.get('update_G_every_n_batches', 1)
    update_F_every_n_batches = train_config.get('update_F_every_n_batches', 1)
    print(f"Generator (G) will be updated every {update_G_every_n_batches} batch(es).")
    print(f"Classifier (F) will be updated every {update_F_every_n_batches} batch(es).")

    print("--- Starting Alternating Training for Clean-Label Backdoor Attack ---")

    for epoch in range(train_config['epochs']):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{train_config['epochs']}")

        for batch_idx, (x_clean, y_true) in progress_bar:
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # ==========================================================
            # == 阶段一: 更新分类器 F_w (Classifier Update)           ==
            # ==========================================================
            if (batch_idx + 1) % update_F_every_n_batches == 0:
                classifier.train()
                generator.eval()

                non_target_mask = (y_true != train_config['target_class'])
                if non_target_mask.any():
                    x_to_poison = x_clean[non_target_mask]

                    with torch.no_grad():
                        x_poisoned, _ = attack_helper.generate_poisoned_sample(x_to_poison)

                    x_batch_for_F = torch.cat([x_clean, x_poisoned], dim=0)
                    y_batch_for_F = torch.cat([y_true, y_true[non_target_mask]], dim=0)
                else:
                    x_batch_for_F = x_clean
                    y_batch_for_F = y_true

                optimizer_F.zero_grad()
                outputs = classifier(x_batch_for_F)
                loss_F = criterion_F(outputs, y_batch_for_F)
                loss_F.backward()
                optimizer_F.step()
            else:
                loss_F = torch.tensor(0.0)  # 如果不更新，损失记为0

            # ==========================================================
            # == 阶段二: 更新触发器生成器 G_θ (Generator Update)     ==
            # ==========================================================
            if (batch_idx + 1) % update_G_every_n_batches == 0:
                classifier.eval()
                generator.train()

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
                loss_G = torch.tensor(0.0)  # 如果不更新，损失记为0

            progress_bar.set_postfix({
                'Loss_F': f'{loss_F.item():.4f}',
                'Loss_G': f'{loss_G.item():.4f}'
            })

        scheduler_F.step()

        # --- 定期评估 ---
        if (epoch + 1) % train_config['eval_every_epochs'] == 0 or (epoch + 1) == train_config['epochs']:
            ba, asr = evaluate(classifier, generator, test_loader, device,
                               train_config['target_class'],
                               train_config['trigger_net'],
                               train_config['injection_strength'])
            print(f"\nEpoch {epoch + 1}: Benign Accuracy (BA) = {ba:.2f}%, Attack Success Rate (ASR) = {asr:.2f}%")

    # --- 保存最终模型 ---
    save_dir = train_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    classifier_to_save = classifier.module if isinstance(classifier, nn.DataParallel) else classifier
    generator_to_save = generator.module if isinstance(generator, nn.DataParallel) else generator

    torch.save(classifier_to_save.state_dict(), os.path.join(save_dir, train_config['classifier_save_name']))
    torch.save(generator_to_save.state_dict(), os.path.join(save_dir, train_config['generator_save_name']))
    print(f"\nFinal models saved to {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alternating training for clean-label backdoor attacks.")
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)