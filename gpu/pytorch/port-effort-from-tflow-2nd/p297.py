import torch
import torch.nn as nn
import helper

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=ToTensor())


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

print("training_data/test_data: ", type(training_data), len(training_data), type(test_data), len(test_data))
print("type: ", type(training_data[0]))
'''
# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# Create dummy input and target tensors (data)
#x = torch.randn(batch_size, n_in)
#y = torch.tensor([[1.0], [0.0], [0.0],
#[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# Create a model
'''

l1=nn.Linear(28*28, 300)
r1=nn.ReLU()
l2=nn.Linear(300, 100)
r2=nn.ReLU()
l3=nn.Linear(100, 30)
'''
flatten = nn.Flatten()
'''

model = nn.Sequential(\
    l1,
    r1,
    l2,
    r2,  
    l3)

print("Model: ", model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(50):
    for (x, y) in enumerate(training_data):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print('epoch: ', epoch,' loss: ', loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

