import torch
import torch.nn as nn
import torch.nn.functional as F

import repackage 
repackage.up()
# two classifiers one for the Background's texture and the other for the object's

class Flatten(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class MNIST_WILDLIFE_BGTXT(nn.Module):

    def __init__(self, num_classes):
        super(MNIST_WILDLIFE_BGTXT, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # self.pool = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3)
        # self.flatten = nn.Sequential(Flatten())
        # self.fc1 = nn.Linear(25*25*10, 120)
        # self.fc2 = nn.Linear(120, 60)
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            Flatten()
        )

        self.output_to_probs = nn.Sequential(
            # nn.Linear(1350, 120),
            # nn.Linear(120, 60),
            # nn.Linear(120, num_classes),
            nn.Linear(1350, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.flatten(x) 
        # x = F.relu(self.fc1(x))#F.leaky_relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.model(x)
        x = self.output_to_probs(x)

        return x 


class MNIST_WILDLIFE_OBTXT(nn.Module):
    def __init__(self, num_classes):
        super(MNIST_WILDLIFE_OBTXT, self).__init__()
        pass



### training and testing pipiline
# import tqdm

#from mnists.dataloader import get_dataloaders


def train(args, model, device, train_data_loader, optimizer, epoch, target_type, max_batches=-1):
    model.train()
    correct = 0
    for batch_idx, data_dict in enumerate(train_data_loader):
        img_tensor = data_dict['ims']
        if target_type == 'shape':
            target = data_dict['labels']
        elif target_type == 'background_texture' or target_type == 'back_text' :
            target = data_dict['back_text']
        elif target_type == 'object texture' or target_type == 'ob_text' or target_type == 'obj_text':
            target = data_dict['obj_text']
        elif target_type == 'bg_category':
            raise NotImplementedError('Background category classificaiton not implemented yet. ToDo: get the labels')
        elif target_type == 'obj_category':
            raise NotImplementedError('Object category classificaiton not implemented yet. ToDo: get the labels')
        else:
            raise ValueError(f"Target {target_type} not an option")
    

        img_tensor, target = img_tensor.to(device), img_tensor.to(device)

        optimizer.zero_grad()  # zero-out parameter gradients

        output = model(img_tensor)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        #gather stats
        pred = output.argmax(dim=1, keepdim=True)   # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        if max_batches > 0 and batch_idx > max_batches:
            break
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
    loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))
    
    train_acc = 100. * correct / len(train_loader.dataset)
    return train_acc, loss