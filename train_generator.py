import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from core.models.resnet import ResNet18
from core.models.generator import UNet
from core.attack import LFDA_U_Attack
from data.cifar10 import get_dataloader


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    # Load data
    train_loader = get_dataloader(
        batch_size=config['generator_training']['batch_size'],
        train=True,
        path=config['dataset']['path'],
        num_workers=config['num_workers']
    )

    # Initialize models
    surrogate_model = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    surrogate_model.load_state_dict(torch.load(config['surrogate_model']['pretrained_path'], map_location=device))
    surrogate_model.eval()
    for param in surrogate_model.parameters():
        param.requires_grad = False

    generator = UNet().to(device)

    # Setup attack logic and optimizer
    attack = LFDA_U_Attack(
        generator=generator,
        surrogate_model=surrogate_model,
        loss_weights=config['generator_training'],
        device=device
    )
    optimizer = optim.Adam(generator.parameters(), lr=config['generator_training']['lr'])

    # Training loop
    print("Starting generator training...")
    for epoch in range(config['generator_training']['epochs']):
        generator.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['generator_training']['epochs']}")

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            x_p, delta_phi = attack.forward_pass(x)
            loss = attack.calculate_loss(x, y, x_p, delta_phi, config['generator_training']['target_class'])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{total_loss / (progress_bar.n + 1):.4f}'})

    # Save the trained generator
    save_dir = config['generator_training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, config['generator_training']['save_name'])
    torch.save(generator.state_dict(), save_path)
    print(f"Generator saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
