from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import torch
from torchvision.utils import make_grid


def save_image_grid(images: list, labels: list, filepath: str, nrow: int = 3):
    """Saves a grid of images with labels."""
    pil_images = [TF.to_pil_image(img.cpu()) for img in images]

    # Add labels to images
    labeled_images = []
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, pil_img in enumerate(pil_images):
        draw = ImageDraw.Draw(pil_img)
        draw.text((5, 5), labels[i], fill="red", font=font)
        labeled_images.append(TF.to_tensor(pil_img))

    grid = make_grid(labeled_images, nrow=nrow, padding=5)
    TF.to_pil_image(grid).save(filepath)
