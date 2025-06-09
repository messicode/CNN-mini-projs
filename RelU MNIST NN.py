import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (Dataset,DataLoader)
from torchvision import datasets
from torchvision.transforms import ToTensor


device=torch.device('cpu')

seed = 42
torch.manual_seed(seed)

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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate=0.01
optimizer=optim.SGD(model.parameters(), lr=learning_rate)


epochs=10

print(f'Hyperparameters used: Number of epochs = {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch}')

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


    print(f'Epoch:{epoch+1}, Loss: {loss_epoch:.4f}, Accuracy: {acc_epoch:.2f}%')