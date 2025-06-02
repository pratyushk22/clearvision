import torch
from models.generator import DummyGenerator  # Replace with actual class later

def load_model():
    model = DummyGenerator()
    
    # Your teammate will replace this with:
    # model.load_state_dict(torch.load('models/gan_generator_epoch10.pth'))

    model.eval()
    return model

def restore_image(model, image_tensor):
    return model(image_tensor)