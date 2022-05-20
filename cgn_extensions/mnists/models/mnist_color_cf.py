import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Flatten(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class RGB_CNN(nn.Module):
    def __init__(self):
        super(RGB_CNN, self).__init__()
        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(3),
        #     nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(7),
        #     Flatten(),
        # )
        # Simple CNN 
        self.model = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            Flatten(),
        )
        
        self.cls = nn.Sequential(
            nn.Linear(3468, 3),
        )

    def forward(self, x):
        return self.cls(self.model(x))

## Utils for training and testing

import os
import argparse
import repackage
repackage.up()

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS

def train(args, model, device, train_loader, optimizer, epoch, target_type, max_batches=-1):
    model.train()
    loss_sum = 0
    train_samples = 0
    for batch_idx, data_dict in enumerate(train_loader):
        data = data_dict['ims']
        train_samples += data.shape[0]
        if target_type == 'shape':
            target = data_dict['labels']
        elif target_type == 'texture':
            target = data_dict['texture_labels']
        elif target_type == 'bg':
            target = data_dict['bg_labels']
        else:
            raise ValueError("Invalid target type")
                
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        mse_loss = nn.MSELoss(reduction='sum')
        loss = mse_loss(output, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        if max_batches > 0 and batch_idx > max_batches:
            break
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()/data.shape[0] ))

    # print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     loss, correct, len(train_loader.dataset),
    #     100. * correct / len(train_loader.dataset)))
    avg_loss = loss_sum / train_samples
    print('Train set: Average epoch loss: {:.4f}\n'.format(avg_loss))
    
    return avg_loss
  

def test(model, device, test_loader, target_type):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data_dict in test_loader:
            data = data_dict['ims']
            if target_type == 'shape':
                target = data_dict['labels']
            elif target_type == 'texture':
                target = data_dict['texture_labels']
            elif target_type == 'bg':
                target = data_dict['bg_labels']
            else:
                raise ValueError("Invalid target type")

            data, target = data.to(device), target.to(device)
            output = model(data)
            mse_loss = nn.MSELoss(reduction='sum')
            test_loss += mse_loss(output, target).item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    test_loss = test_loss / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss

