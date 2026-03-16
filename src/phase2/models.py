import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeStateNet(nn.Module):
    """
    Lightweight CNN for eye state (Open/Closed) classification.
    Input: (1, 24, 24) or (1, 32, 32) grayscale eye crops.
    """
    def __init__(self):
        super(EyeStateNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Output size after 3 pools (32 -> 16 -> 8 -> 4)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2) # [Closed, Open]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GazeMLP(nn.Module):
    """
    MLP for Gaze Zone Classification (11 zones).
    Input: (Yaw, Pitch, Roll) + iris offsets + eye/mouth features (approx 20 dims).
    """
    def __init__(self, input_dim=19, num_classes=11):
        super(GazeMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Test architectures
    eye_model = EyeStateNet()
    gaze_model = GazeMLP()
    print(f"Eye Model Params: {sum(p.numel() for p in eye_model.parameters())}")
    print(f"Gaze Model Params: {sum(p.numel() for p in gaze_model.parameters())}")
