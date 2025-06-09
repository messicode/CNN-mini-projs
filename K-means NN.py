# *****************************************************************************
# *****************************************************************************
# k-means using a neural network
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
        self.flatten = nn.Flatten() #!!! Fill in !!!#
        self.centers = nn.Parameter(torch.rand(10, 28 * 28)) #!!! Fill in !!!# Use nn.Parameter
        self.softmax = nn.Softmax(dim=1) #!!! Fill in !!!# Along dimension 1

    def forward(self, x):
        x0 = self.flatten(x)
        x = torch.matmul(x0, self.centers.t()) - 0.5 * torch.sum(self.centers ** 2, 1).flatten() #!!! Fill in !!!# Get surrogate for -distance
        x = 20*x
        x = self.softmax(x)
        x = torch.matmul(x, self.centers) #!!! Fill in !!!# Get center
        x = x-x0 #!!! Fill in !!!# Get error
        return x

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
        loss = loss_fn(pred, torch.zeros_like(pred)) #!!! Fill in !!!# Compare the error return by the model to 0

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
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, torch.zeros_like(pred)).item() #!!! Fill in !!!# Compare the error return by the model to 0
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n {size} {num_batches}")


# *****************************************************************************
# Optimization prameters and initialization
# *****************************************************************************
basic_train_dataloader = DataLoader(training_data, batch_size=1)
training_size = len(basic_train_dataloader.dataset)

loss_fn = nn.MSELoss() #!!! Fill in !!!# Mean squared error
learning_rate = 4.5 #!!! Fill in !!!#
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initialization:
with torch.no_grad():
  # model.centers[:] = torch.zeros((10,28*28)).to(device)
  i=0
  for X, y in basic_train_dataloader:
    if y==i: # You could try randomizing whether you pick this point or not.
      model.centers[i, :] = X.flatten() #!!! Fill in !!!# Set the i-th center to this X
      i += 1
    if i==10:
      break

# *****************************************************************************
# Standard training epochs
# *****************************************************************************
print(model)
print("Training model...")
epochs = 10 #!!! Fill in !!!#
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# *****************************************************************************
# Building the confusion matrix
# *****************************************************************************
print("Computing the confusion matrix...")
C = model.centers.detach().cpu() # Extract the centers from the model
counts = torch.zeros(10, 10) #!!! Fill in !!!# Initialize counts matrix

with torch.no_grad():
  for X, y in basic_train_dataloader:
    best_distance = 1e16 # Very large number for initialization
    best_index = 0 # Arbitrary index for initialization
    for j in range(10):
      dist = torch.norm(X.flatten() - C[j]) #!!! Fill in !!!# Calculate distance of X from center j
      if dist < best_distance: #!!! Fill in !!!# Determine condition to update the distance and index
        best_distance = dist #!!! Fill in !!!# Update the distance
        best_index = j #!!! Fill in !!!# Update the index
    counts[y.item() , best_index] += 1 #!!! Fill in !!!# Update the counts at the (label, cluster) index

print(counts.int())
# print(counts.sum().item())

# *****************************************************************************
# Displaying the centers
# *****************************************************************************
print("Cluster centers:")
for j in range(10):
  print(f"Cluster {j}")
  q = C[j].view(28,28) #!!! Fill in !!!# Grab center j
  plt.imshow(q, cmap="gray") #!!! Fill in !!!# Display center j as an image
  plt.show()