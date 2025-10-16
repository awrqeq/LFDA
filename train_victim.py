import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# --- ADDED: Environment setup for torch.hub ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- END ADDITION ---

from core.models.resnet import ResNet18
from core.models.generator import TriggerNet
from data.cifar10 import get_dataloader
from core.utils import freq_utils


# --- MODIFIED: evaluate function updated for multi-GPU ---
def evaluate(victim_model, generator, test_loader, device, target_class, trigger_net_config, injection_strength):
    victim_model.eval()
    generator.eval()

    total_correct_clean = 0
    total_correct_poisoned = 0
    total_poisoned_samples = 0
    total_samples = 0

    patch_size = trigger_net_config['patch_size']
    start_row = trigger_net_config['start_row']
    start_col = trigger_net_config['start_col']

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating BA", leave=False):
            x, y = x.to(device), y.to(device)
            outputs = victim_model(x)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += y.size(0)
            total_correct_clean += (predicted == y).sum().item()

        ba = 100 * total_correct_clean / total_samples if total_samples > 0 else 0

        for x, y in tqdm(test_loader, desc="Evaluating ASR", leave=False):
            x, y = x.to(device), y.to(device)
            non_target_mask = (y != target_class)
            if not non_target_mask.any():
                continue

            x_to_poison = x[non_target_mask]

            x_freq = freq_utils.to_freq(x_to_poison)
            freq_patch = freq_utils.extract_freq_patch_and_reshape(x_freq, patch_size, start_row, start_col)
            delta_patch = generator(freq_patch)
            poisoned_freq = freq_utils.reshape_and_insert_freq_patch(x_freq, delta_patch, patch_size, start_row,
                                                                     start_col, injection_strength)
            x_p = freq_utils.to_spatial(poisoned_freq)
            x_p = torch.clamp(x_p, 0, 1)

            outputs = victim_model(x_p)
            _, predicted = torch.max(outputs.data, 1)

            total_poisoned_samples += x_to_poison.size(0)
            total_correct_poisoned += (predicted == target_class).sum().item()

    asr = 100 * total_correct_poisoned / total_poisoned_samples if total_poisoned_samples > 0 else 0

    return ba, asr


# --- END MODIFICATION ---

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- ADDED: Device setup logic ---
    if torch.cuda.is_available() and config.get('device_ids'):
        device_ids = config['device_ids']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        if len(device_ids) > 0:
            device = torch.device("cuda:0")
            print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            device_ids = []
            device = torch.device("cpu")
            print("No GPUs specified, using CPU.")
    else:
        device_ids = []
        device = torch.device("cpu")
        print("CUDA not available or no GPUs specified, using CPU.")
    # --- END ADDITION ---

    torch.manual_seed(config['seed'])

    train_loader = get_dataloader(config['victim_training']['batch_size'], True, config['dataset']['path'],
                                  config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- MODIFIED: Model initialization for multi-GPU ---
    victim_model = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    generator = TriggerNet().to(device)
    generator.load_state_dict(torch.load(config['victim_training']['generator_path'], map_location=device))

    if len(device_ids) > 1:
        victim_model = nn.DataParallel(victim_model)
        generator = nn.DataParallel(generator)
    # --- END MODIFICATION ---

    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False

    params_to_optimize = victim_model.module.parameters() if isinstance(victim_model,
                                                                        nn.DataParallel) else victim_model.parameters()
    optimizer = optim.SGD(params_to_optimize, lr=config['victim_training']['lr'],
                          momentum=config['victim_training']['momentum'],
                          weight_decay=config['victim_training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['victim_training']['epochs'])
    criterion = nn.CrossEntropyLoss()

    trigger_net_config = config['generator_training']['trigger_net']
    injection_strength = config['generator_training']['injection_strength']
    target_class = config['victim_training']['target_class']
    poison_rate = config['victim_training']['poison_rate']

    print("--- Starting Victim Model Training (NFF) ---")
    for epoch in range(config['victim_training']['epochs']):
        model_to_train = victim_model.module if isinstance(victim_model, nn.DataParallel) else victim_model
        model_to_train.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['victim_training']['epochs']}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            #
            # ####################################################################
            # ### 核心修改: 将污染逻辑改为只污染目标类别的样本             ###
            # ####################################################################
            #
            # 1. 找到当前批次中所有属于目标类别的样本的索引
            target_class_indices = (y == target_class).nonzero(as_tuple=True)[0]

            # 2. 只有当批次中存在目标类别样本时，才进行污染
            if target_class_indices.numel() > 0:
                # 3. 根据poison_rate，计算需要污染的目标类别样本数量
                num_to_poison = int(poison_rate * target_class_indices.numel())

                if num_to_poison > 0:
                    # 4. 从目标类别样本中，随机选择一部分进行污染
                    perm = torch.randperm(target_class_indices.numel())
                    poison_indices = target_class_indices[perm[:num_to_poison]]

                    x_to_poison = x[poison_indices]

                    with torch.no_grad():
                        x_freq = freq_utils.to_freq(x_to_poison)
                        freq_patch = freq_utils.extract_freq_patch_and_reshape(x_freq, **trigger_net_config)
                        delta_patch = generator(freq_patch)
                        poisoned_freq = freq_utils.reshape_and_insert_freq_patch(x_freq, delta_patch,
                                                                                 strength=injection_strength,
                                                                                 **trigger_net_config)
                        x_p = freq_utils.to_spatial(poisoned_freq)
                        x_p = torch.clamp(x_p, 0, 1)

                    # 5. 将污染后的样本放回原位
                    x[poison_indices] = x_p
            # ####################################################################

            optimizer.zero_grad()
            outputs = victim_model(x)
            loss = criterion(outputs, y)

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        scheduler.step()

        if (epoch + 1) % config['victim_training']['eval_every_epochs'] == 0:
            ba, asr = evaluate(victim_model, generator, test_loader, device, target_class, trigger_net_config,
                               injection_strength)
            print(f"\nEpoch {epoch + 1}: Benign Accuracy (BA) = {ba:.2f}%, Attack Success Rate (ASR) = {asr:.2f}%")

    print("\n--- Final Evaluation ---")
    ba, asr = evaluate(victim_model, generator, test_loader, device, target_class, trigger_net_config,
                       injection_strength)
    print(f"Final Benign Accuracy (BA): {ba:.2f}%")
    print(f"Final Attack Success Rate (ASR): {asr:.2f}%")

    save_dir = config['victim_training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, config['victim_training']['save_name'])

    model_to_save = victim_model.module if isinstance(victim_model, nn.DataParallel) else victim_model
    torch.save(model_to_save.state_dict(), save_path)
    print(f"Victim model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)