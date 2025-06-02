import torch.nn as nn

class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity() 

    def forward(self, x):
        return x