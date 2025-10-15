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
from core.models.generator import UNet
from core.attack import LFDA_U_Attack
from data.cifar10 import get_dataloader


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- MODIFIED: Device setup for multi-GPU ---
    if torch.cuda.is_available() and config.get('device_ids'):
        device_ids = config['device_ids']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        if len(device_ids) > 0:
            device = torch.device(f"cuda:0")  # Primary device
            print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            device = torch.device("cpu")
            print("No GPUs specified, using CPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    # --- END MODIFICATION ---

    torch.manual_seed(config['seed'])

    train_loader = get_dataloader(config['generator_training']['batch_size'], True, config['dataset']['path'],
                                  config['num_workers'])

    # --- MODIFIED: Model initialization for multi-GPU ---
    surrogate_model = ResNet18(num_classes=config['dataset']['num_classes'])
    surrogate_model.load_state_dict(torch.load(config['surrogate_model']['pretrained_path'], map_location='cpu'))
    surrogate_model = surrogate_model.to(device)

    generator = UNet().to(device)

    if len(device_ids) > 1:
        surrogate_model = nn.DataParallel(surrogate_model)
        generator = nn.DataParallel(generator)
    # --- END MODIFICATION ---

    surrogate_model.eval()

    attack = LFDA_U_Attack(generator=generator, surrogate_model=surrogate_model,
                           loss_weights=config['generator_training'], device=device)

    # --- MODIFICATION: Ensure optimizer gets params from the original model if wrapped ---
    params_to_optimize = generator.module.parameters() if isinstance(generator,
                                                                     nn.DataParallel) else generator.parameters()
    optimizer = optim.Adam(params_to_optimize, lr=config['generator_training']['lr'])
    # --- END MODIFICATION ---

    print("Starting generator training...")
    for epoch in range(config['generator_training']['epochs']):
        # Use .module to access original model's train() method if wrapped
        model_to_train = generator.module if isinstance(generator, nn.DataParallel) else generator
        model_to_train.train()

        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['generator_training']['epochs']}")

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            x_p, delta_phi = attack.forward_pass(x)
            loss = attack.calculate_loss(x, y, x_p, delta_phi, config['generator_training']['target_class'])

            # --- MODIFICATION for DataParallel: average loss if multiple GPUs ---
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            # --- END MODIFICATION ---

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{total_loss / (progress_bar.n + 1):.4f}'})

    save_dir = config['generator_training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, config['generator_training']['save_name'])

    # --- MODIFICATION: Save the state_dict from the original model ---
    model_to_save = generator.module if isinstance(generator, nn.DataParallel) else generator
    torch.save(model_to_save.state_dict(), save_path)
    # --- END MODIFICATION ---

    print(f"Generator saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)