import torchvision.transforms as transforms
from PIL import Image
import torch

def preprocess_image(image_path):
    """
    Load an image from disk, resize to 256×256, and convert to a tensor in [-1, +1].
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])  
    ])
    tensor = transform(image).unsqueeze(0)  
    return tensor

def postprocess_tensor(tensor):
    """
    Convert a tensor in [-1, +1] back to a PIL Image.
    """

    tensor = tensor.clamp(-1, 1)

    tensor = (tensor + 1.0) / 2.0

    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = (array * 255).astype("uint8")
    return Image.fromarray(array)
