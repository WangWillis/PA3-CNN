################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Jenny Hamer #
# Filename: baseline_cnn.py
# # Description: 
# 
# This file contains the starter code for the baseline architecture you will use
# to get a little practice with PyTorch and compare the results of with your 
# improved architecture. 
#
# Be sure to fill in the code in the areas marked #TODO.
################################################################################


# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os

class Impractical_Loss(torch.nn.Module):
    def __init__(self, weight=None, pen=1e-1):
        super(Impractical_Loss, self).__init__()
        self.weight = weight
        self.pen = pen

    def forward(self, y, t):
        eps = 1e-16
        # diff = torch.abs(t-y)
        # c = -(t*torch.log(y+eps)+self.pen*(1-t)*torch.log(1-y+eps))
        c = -(t*torch.log(y+eps)+self.pen*(1-t)*torch.log(1-y+eps))
        # c = -(self.weight*t*torch.log(y+eps)+y*(1-t)*torch.log(1-y+eps))
        # a = t*torch.log(y+eps)
        # b = (1-t)*torch.log(1-y+eps)
        # c = -(a+b*torch.abs(b))
        if (self.weight is not None):
            c *= self.weight

        return torch.sum(c)/(1+torch.sum(t))

KERNEL_SIZE = 3
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=False):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c).cuda()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=KERNEL_SIZE, stride=stride, padding=1).cuda()
        # self.bn1 = nn.BatchNorm2d(out_c).cuda()

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=KERNEL_SIZE, stride=stride, padding=1).cuda()
        self.bn2 = nn.BatchNorm2d(out_c).cuda()

        self.downsample = downsample
        if (self.downsample):
            self.conv_down = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride).cuda()
            self.bn_down = nn.BatchNorm2d(out_c).cuda()

    def forward(self, batch):
        batch = self.bn1(batch)

        org = batch
        if (self.downsample):
            org = self.conv_down(batch)
            org = self.bn_down(org)

        res = self.conv1(batch)
        # res = self.bn1(res)
        res = func.relu(res)

        res = self.bn2(res)
        res = self.conv2(res)
        out = org+res
        out = func.relu(out)

        return out

class ResLayer(nn.Module):
    def __init__(self, in_c, out_c, blocks=4, pool=True):
        super(ResLayer, self).__init__()
        self.blocks = []
        self.pool = pool

        self.blocks.append(ResBlock(in_c, out_c, downsample=True))
        for _ in range(1, blocks):
            self.blocks.append(ResBlock(out_c, out_c))
        if (self.pool):
            self.pool = nn.MaxPool2d(kernel_size=3, stride=3).cuda()

    def forward(self, batch):
        res = batch
        for block in self.blocks:
            res = block(res)
        if (self.pool):
            res = self.pool(res)
        return res

class ResCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(ResCNN, self).__init__()

        S = 3
        FC1_IN = (2**(S+1))*(256**2)

        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 2**S, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(2**S)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResLayer(2**S, 2**S, blocks=2, pool=False)
        self.layer2 = ResLayer(2**S, 2**(S+1), blocks=4, pool=False)
        
        self.bnf = nn.BatchNorm1d(FC1_IN)
        self.fc1 = nn.Linear(FC1_IN, num_classes)

        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
                torch_init.xavier_normal_(m.weight)

    def forward(self, batch):
        out = self.bn1(batch)
        out = self.conv1(out)
        # out = self.bn1(out)
        out = func.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = out.view(-1, self.num_flat_features(out))

        out = self.bnf(out)
        out = self.fc1(out)

        return func.sigmoid(out)

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

    def visualize_filters(self):
        weight = self.conv1.weight.data.numpy()
        plt.imshow(weight[0,...])
        weight = self.conv2.weight.data.numpy()
        plt.imshow(weight[0,...])
        weight = self.conv3.weight.data.numpy()
        plt.imshow(weight[0,...])

def getResults(preds, targs, thresh = 0.5):
    preds = preds.cpu().detach().numpy()
    targs = targs.cpu().detach().numpy()

    preds[preds < thresh] = 0.
    preds[preds >= thresh] = 1.

    tp = np.sum(np.logical_and(preds == 1, targs == 1))
    fp = np.sum(np.logical_and(preds == 1, targs == 0))
    fn = np.sum(np.logical_and(preds == 0, targs == 1))
    tn = np.sum(np.logical_and(preds == 0, targs == 0))

    return tp, tn, fp, fn

# Setup: initialize the hyperparameters/variables
num_epochs = 8           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 1e-4
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
# transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

def main():
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                     p_val=p_val, p_test=p_test,
                                     shuffle=True, show_sample=False, 
                                     extras=extras)

    # Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
    model = ResCNN()
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    WEIGHTS = torch.tensor([12.84306987, 55.5324418, 11.7501572, 7.83946301, 26.91956783, 24.54465849, 117.64952781, 30.0670421, 33.64945978, 67.95151515, 61.70610897, 91.7512275, 45.91318591, 671.37724551]).to(computing_device)/10
    # criterion = nn.BCEWithLogitsLoss(weight=WEIGHTS)
    criterion = Impractical_Loss(weight=WEIGHTS, pen=0.4)
    # criterion = Impractical_Loss(weight=None, pen=0.5)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    avg_train_acc = []

    val_loss = []
    val_acc = [] 

    best_loss = float('inf')

    
    # Begin training procedure
    for epoch in range(num_epochs):
        sum_prec = 0.
        sum_recall = 0.
        sum_brc = 0.

        N = 200
        N_minibatch_loss = 0.0    
        n_train_acc = 0.0

        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)
            
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            
            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += loss
            tp, tn, fp, fn = getResults(outputs, labels)
            n_train_acc += (tp+tn)/(tp+tn+fp+fn)
            
            if minibatch_count != 0 and minibatch_count % N == 0:
                # Print the loss averaged over the last N mini-batches    
                N_minibatch_loss /= N
                n_train_acc /= N
                avg_train_acc.append(float(n_train_acc))

                images, labels = next(iter(val_loader))

                images, labels = images.to(computing_device), labels.to(computing_device)
                # Perform the forward pass through the network and compute the loss
                outputs = model(images)
                loss = criterion(outputs, labels)

                tp, tn, fp, fn = getResults(outputs, labels)
                v_acc = (tp+tn)/(tp+tn+fp+fn)

                precision = 0
                recall = 0
                brc = 0
                if (tp+fn != 0):
                    recall = tp/(tp+fn)
                if (fp+tp != 0):
                    precision = tp/(tp+fp)
                brc = (precision+recall)/2

                sum_prec += precision
                sum_recall += recall
                sum_brc += brc
                val_loss.append(float(loss.item()))
                val_acc.append(float(v_acc))

                if (loss < best_loss):
                    torch.save(model.state_dict(), 'best_res_model_test2.pt')
                    best_loss = loss.item()
                 
                print('Epoch %d, minibatch %d average loss: %.3f, average acc: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss, n_train_acc))
                print('Epoch %d, validation loss: %.3f, validation acc: %.3f' % (epoch+1, loss.item(), v_acc))
                print('Precision: %.3f, Recall: %.3f, BRC:%.3f' % (precision, recall, brc))
                print('tp: %d, tn: %d, fp: %d, fn: %d' % (tp, tn, fp, fn))
                
                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(float(N_minibatch_loss))
                N_minibatch_loss = 0.0
                n_train_acc = 0.0

        print('Precision: %.3f, Recall: %.3f, BRC:%.3f' % (N*sum_prec/5000, N*sum_recall/5000, N*sum_brc/5000))
        print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")

    train_data = np.array([avg_minibatch_loss, avg_train_acc, val_loss, val_acc]) 
    np.save('res_data_test2.npy', train_data)

if __name__ == '__main__':
    main()
