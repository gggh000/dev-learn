import torch
import torch.nn as nn
import helper
import sys
import time
import numpy as np
DEBUG=0
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
CONFIG_EPOCHS=3
CONFIG_BATCH_SIZE=32
for i in sys.argv:
    print("Processing ", i)
    try:
        if re.search("epochs=", i):
            CONFIG_EPOCHS=int(i.split('=')[1])

        if re.search("batch_size=", i):
            CONFIG_BATCH_SIZE=int(i.split('=')[1])

    except Exception as msg:
        print("No argument provided, default values will be used.")

print("---")
print("epochs: ", CONFIG_EPOCHS,": ", end=' ', flush=True)
#print("batch_size: ", CONFIG_BATCH_SIZE)

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

trainloader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)

print("training_data/test_data: ", type(training_data), len(training_data), type(test_data), len(test_data))
print("type: ", type(training_data[0]))

print("trainloader: ", type(trainloader))
print("testloader:  ", type(testloader))

f1=nn.Flatten()
l1=nn.Linear(28*28, 300)
r1=nn.ReLU()
l2=nn.Linear(300, 100)
r2=nn.ReLU()
l3=nn.Linear(100, 10)
s3=nn.Softmax()
model = nn.Sequential(\
    f1,
    l1,
    r1,
    l2,
    r2,  
    l3, 
    s3, \
)

print("Model: ", model)

#print("model: layer0: ", model[0], model[0].weight)
print("l1 info: ", l1, l1.weight.shape)
print("l2 info: ", l2, l2.weight.shape)
print("l3 info: ", l3, l3.weight.shape)

#criterion = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


i=0
for epoch in range(CONFIG_EPOCHS):
    print('epoch/i: ', epoch, i)
    j=0
    for batch in trainloader:
    
        imgs, lbls = batch

        if j == 0:
            bypass_dots=int(len(training_data)/len(lbls)/80)

            if DEBUG:
                print("bypass_dots quantity: ", bypass_dots)

        if DEBUG:
            print("batch: ", type(batch), ", ", len(batch))
            print("imgs: ", type(imgs), ", ", len(imgs), imgs.shape)
            print("lbls: ", type(lbls), ", ", len(lbls), lbls.shape)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(imgs)

        if DEBUG:
            print("y_pred: ", type(y_pred), y_pred.shape)
            print("lbls:   ", type(lbls), lbls.shape)

        # Compute and print loss
        loss = criterion(y_pred, lbls)

        if DEBUG:
            print('epoch/batch: ', epoch, i,' loss: ', loss.item())

        if j%bypass_dots == 0:
            print(".", end='', flush=True)
    
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()
        j+=1
    i+=1

print("Testing...")

i=0

print(len(testloader), type(testloader))

i=0
#for batch in testloader:
#    imgs, lbls = batch
for imgs, lbls in testloader:
    print("---", i, "---")
    print("imgs: ", imgs.shape)
    print("lbls: ", lbls.shape)
    i+=1
y_pred=model(imgs)
print("y_pred: ", y_pred.shape, type(y_pred))
