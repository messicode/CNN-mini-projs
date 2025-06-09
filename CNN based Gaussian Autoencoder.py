# *****************************************************************************
# *****************************************************************************
# Gaussian Autoencoder
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# Preamble and dataset loading, based on PyTorch tutorial
# *****************************************************************************
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(device)
print(f"Using {device} device")

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64 #!!! Fill in !!!#

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# *****************************************************************************
# Building the neural network
# *****************************************************************************
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential( # Same as HW6's CNN
            nn.Conv2d(1,20,4,1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.3),
            nn.Conv2d(20,20,4,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(20),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(0.3),
            nn.Linear(250, 64)
        )
        self.Decoder = nn.Sequential( #!!! Fill in !!!# "Invert" the encoder
            nn.Linear(64,360),
            nn.ReLU(),
            nn.BatchNorm1d(360),
            nn.Dropout(0.3),
            nn.Linear(360,720),
            nn.ReLU(),
            nn.Unflatten(1, torch.Size([20, 6, 6])),

            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=1, padding=1,output_padding=0),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x, enc_mode = 1):
        z = self.Encoder(x) #!!! Fill in !!! Encoder, giving embedding
        z2 = enc_mode*z + (2-enc_mode)*torch.randn(z.shape) # Adding noise
        f = self.Decoder(z2) #!!! Fill in !!! Decoder, giving reconstructed x
        e = f-x #!!! Fill in !!! Reconstruction error
        e = self.flatten(e)
        e = torch.cat([z, e], dim=1) #!!! Fill in !!! Concatenate embedding and reconstruction error
        return e

model = NeuralNetwork().to(device)

# *****************************************************************************
# Train and test loops
# *****************************************************************************
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred[:, 64:], torch.zeros_like(pred[:, 64:])) #!!! Fill in !!!

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred[:, 64:],torch.zeros_like(pred[:, 64:]) ).item() #!!! Fill in !!!
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n {size} {num_batches}")

# *****************************************************************************
# Optimization prameters and initialization
# *****************************************************************************
loss_fn =  nn.MSELoss() #!!! Fill in !!!
learning_rate = 0.01 #!!! Fill in !!!#
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# *****************************************************************************
# Standard training epochs
# *****************************************************************************
print(model)
print("Training model...")
epochs = 20 #!!! Fill in !!!#
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# *****************************************************************************
# Generating new images using the learned autoencoder
# *****************************************************************************
for s in range(2):
    x = model(torch.randn(1, 1, 28, 28).to(device), enc_mode=0) #!!! Fill in !!! Generate a new image by calling the model with a 0 input and argument enc_mode = 0
    imgX = x[0, 64:].reshape(28,28).detach().to("cpu") #!!! Fill in !!! Extract the image part of x
    plt.imshow(imgX)
    plt.show()
