import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self):
        super(FusionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 128),  # ✅ changed from 256 → 5
            nn.ReLU(),
            nn.Linear(128, 3)   # Assuming output is (x, y, z)
        )

    def forward(self, x):
        return self.fc(x)
