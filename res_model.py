################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Jenny Hamer #
# Filename: baseline_cnn.py
# 
# Description: 
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
        eps = 1e-8
        diff = torch.abs(t-y)
        c = -diff*(t*torch.log(y+eps)+self.pen*(1-t)*torch.log(1-y+eps))
        if (self.weight is not None):
            c *= self.weight

        return torch.sum(c)

KERNEL_SIZE = 3
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=KERNEL_SIZE, stride=stride, padding=1).cuda()
        self.bn1 = nn.BatchNorm2d(out_c).cuda()

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=KERNEL_SIZE, stride=stride, padding=1).cuda()
        self.bn2 = nn.BatchNorm2d(out_c).cuda()

        self.downsample = downsample
        if (self.downsample):
            self.conv_down = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride).cuda()
            self.bn_down = nn.BatchNorm2d(out_c).cuda()

    def forward(self, batch):
        org = batch
        if (self.downsample):
            org = self.conv_down(batch)
            org = self.bn_down(org)

        res = self.conv1(batch)
        res = self.bn1(res)
        res = func.relu(res)

        res = self.conv2(res)
        res = self.bn2(res)
        out = org+res
        out = func.relu(out)

        return out

class ResLayer(nn.Module):
    def __init__(self, in_c, out_c, blocks=4):
        super(ResLayer, self).__init__()
        self.blocks = []

        self.blocks.append(ResBlock(in_c, out_c, downsample=True))
        for _ in range(1, blocks):
            self.blocks.append(ResBlock(out_c, out_c))

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3).cuda()

    def forward(self, batch):
        res = batch
        for block in self.blocks:
            res = block(res)
        res = self.pool(res)
        return res

class ResCNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    Consists of three Conv2d layers, followed by one 4x4 max-pooling layer, 
    and 2 fully-connected (FC) layers:
    
    conv1 -> conv2 -> conv3 -> maxpool -> fc1 -> fc2 (outputs)
    
    Make note: 
    - Inputs are expected to be grayscale images (how many channels does this imply?)
    - The Conv2d layer uses a stride of 1 and 0 padding by default
    """
    
    def __init__(self, num_classes=14):
        super(ResCNN, self).__init__()

        FC1_IN = 32*6*6

        self.layer1 = ResLayer(1, 4, blocks=3)
        self.layer2 = ResLayer(4, 8, blocks=4)
        self.layer3 = ResLayer(8, 16, blocks=6)
        self.layer4 = ResLayer(16, 32, blocks=3)

        
        # Define 2 fully connected layers:
        self.fc1 = nn.Linear(FC1_IN, num_classes)

        #TODO: Output layer: what should out_features be?
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
                torch_init.xavier_normal_(m.weight)

    def create_layer(self, in_c, out_c, num_blocks=1):
        blocks = []

        blocks.append(ResBlock(in_c, out_c, downsample=True))
        for _ in range(1, num_blocks):
            blocks.append(ResBlock(out_c, out_c))

        blocks.append(nn.MaxPool2d(kernel_size=3, stride=3))
        return nn.Sequential(*blocks)

    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying 
        non-linearities after each layer.
        
        Note that this function *needs* to be called "forward" for PyTorch to 
        automagically perform the forward pass. 
        
        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """
        # Apply first convolution, followed by ReLU non-linearity; 
        # use batch-normalization on its outputs

        out = self.layer1(batch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(-1, self.num_flat_features(out))

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
num_epochs = 1           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 1e-3 
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])


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

    WEIGHTS = torch.tensor([12.84306987, 55.5324418, 11.7501572, 7.83946301, 26.91956783, 24.54465849, 117.64952781, 30.0670421, 33.64945978, 67.95151515, 61.70610897, 91.7512275, 45.91318591, 671.37724551]).to(computing_device)
    # criterion = nn.BCEWithLogitsLoss(weight=WEIGHTS) #TODO - loss criteria are defined in the torch.nn package
    # criterion = Impractical_Loss(weight=WEIGHTS, pen=0.1) #TODO - loss criteria are defined in the torch.nn package
    criterion = Impractical_Loss(weight=None, pen=0.1) #TODO - loss criteria are defined in the torch.nn package

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    avg_train_acc = []

    val_loss = []
    val_acc = [] 

    best_loss = float('inf')

    # Begin training procedure
    for epoch in range(num_epochs):
        N = 50
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
            
            if minibatch_count % N == 0:    
                # Print the loss averaged over the last N mini-batches    
                N_minibatch_loss /= N
                n_train_acc /= N
                avg_train_acc.append(n_train_acc)

                images, labels = next(iter(val_loader))

                images, labels = images.to(computing_device), labels.to(computing_device)
                # Perform the forward pass through the network and compute the loss
                outputs = model(images)
                loss = criterion(outputs, labels)

                tp, tn, fp, fn = getResults(outputs, labels)
                print(tp, tn, fp, fn)
                v_acc = (tp+tn)/(tp+tn+fp+fn)

                val_loss.append(loss.item())
                val_acc.append(v_acc)

                if (loss < best_loss):
                    torch.save(model.state_dict(), 'best_res_model.pt')
                    best_loss = loss.item()
                 
                print('Epoch %d, average minibatch %d loss: %.3f, average acc: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss, n_train_acc))
                print('Epoch %d, validation loss: %.3f, validation acc: %.3f' % (epoch+1, loss.item(), v_acc))
                
                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0
                n_train_acc = 0.0

        print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")

    train_data = np.array([avg_minibatch_loss, avg_train_acc, val_loss, val_acc]) 
    np.save('res_data.npy', train_data)

if __name__ == '__main__':
    main()
