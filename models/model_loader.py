import torch
import torch.nn as nn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RestorationNet(nn.Module):
    def __init__(self):
        super(RestorationNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
      
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),   
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_model():
    model = RestorationNet()
    model_path = os.path.join(BASE_DIR, "restoration_model.pth")
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Restoration model loaded successfully.")
    except Exception as e:
        print("Error loading restoration model:", e)
    model.eval()
    return model

def restore_image(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)
