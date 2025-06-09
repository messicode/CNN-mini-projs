class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1),  # Input to 20 channels
            nn.ReLU(),
            nn.BatchNorm2d(20),

            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2),  # Map 20 to 20 channels
            nn.ReLU(),
            nn.BatchNorm2d(20),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Flatten(),
            nn.Linear(20 * 5 * 5, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(p=dropout_probability),
            nn.Linear(250, 10)
        )

    def forward(self, x):
        return self.model(x)