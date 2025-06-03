from .unet_generator import UNetGenerator
import torch
import torch.nn as nn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model():
    model = UNetGenerator()
    model_path = os.path.join(BASE_DIR, "unet_model.pth")
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("U-Net model loaded successfully.")
    except Exception as e:
        print("Error loading restoration model:", e)

    if not any(p.requires_grad for p in model.parameters()):
        print("Warning: Model parameters are not loaded or all frozen.")
    model.eval()
    return model

def restore_image(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)