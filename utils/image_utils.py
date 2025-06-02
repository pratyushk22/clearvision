from PIL import Image
import torchvision.transforms as T

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)  

def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor * 0.5 + 0.5 
    return T.ToPILImage()(tensor)