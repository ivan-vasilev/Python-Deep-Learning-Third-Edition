print("Classifying MNIST with a fully-connected PyTorch network with one hidden layer")

from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
validation_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_data,
    batch_size=100,
    shuffle=True)

validation_loader = DataLoader(
    validation_data,
    batch_size=100,
    shuffle=True)

import torch

torch.manual_seed(1234)

hidden_units = 100
classes = 10

net = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, hidden_units),
    torch.nn.BatchNorm1d(hidden_units),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_units, classes),
)

cost_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

epochs = 20

for epoch in range(epochs):
    train_loss = 0

    for i, (inputs, targets) in enumerate(train_loader):
        # Flatten 28x28 images into a 784 long vector
        inputs = inputs.view(inputs.shape[0], -1)

        optimizer.zero_grad()  # Zero the gradient
        out = net(inputs)  # Forward pass
        loss = cost_func(out, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Weight updates

        train_loss += loss.item() * inputs.size(0)  # Aggregate loss

    train_loss /= len(train_loader.dataset)
    print('Epoch %d, Loss: %.4f' % (epoch + 1, train_loss))

net.eval()  # set network for evaluation
validation_loss = correct = 0
for inputs, target in validation_loader:
    # Flatten 28x28 images into a 784 long vector
    inputs = inputs.view(inputs.shape[0], -1)

    out = net(inputs)  # Forward pass
    loss = cost_func(out, target)  # Compute loss

    # update running validation loss and accuracy
    validation_loss += loss.item() * inputs.size(0)
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

correct = 100 * correct / len(validation_loader.dataset)
validation_loss /= len(validation_loader.dataset)
print('Accuracy: %.1f, Validation loss: %.4f' % (correct, validation_loss))
