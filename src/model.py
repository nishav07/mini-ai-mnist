from torch import nn

class MNISTCNN(nn.Module):
    """
    A small, reliable CNN for MNIST.
    Input: 1x28x28
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
