import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (Dataset,DataLoader)
from torchvision import datasets
from torchvision.transforms import ToTensor

seed = 42
torch.manual_seed(seed)

device=torch.device('cpu')

training_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch = 64
train_loader = DataLoader(training_data, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)
dropout_probability = 0.4

# Part A, C and D
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

model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate=0.05
weightDecay=0.0005
optimizer=optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weightDecay)

epochs=10


print(f'Hyperparameters used: Number of epochs = {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch},Weight Decay: {weightDecay},Dropout Probability (added just before last linear layer):{dropout_probability}')

# Part B

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*data.size(0)
        _, pred = torch.max(output.data,1)
        total += target.size(0)
        correct += (pred==target).sum().item()

    loss_epoch = running_loss/len(train_loader.dataset)
    acc_epoch = 100*correct/total


    print(f'Epoch:{epoch+1}, Loss: {loss_epoch:.2f}, Accuracy: {acc_epoch:.2f}%')
