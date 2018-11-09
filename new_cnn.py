################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Jenny Hamer
#
# Filename: new_cnn.py
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



class BasicCNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    Consists of three Conv2d layers, followed by one 4x4 max-pooling layer, 
    and 2 fully-connected (FC) layers:
    
    conv1 -> conv2 -> conv3 -> maxpool -> fc1 -> fc2 (outputs)
    
    Make note: 
    - Inputs are expected to be grayscale images (how many channels does this imply?)
    - The Conv2d layer uses a stride of 1 and 0 padding by default
    """
    
    def __init__(self):
        super(BasicCNN, self).__init__()

        CONV1_IN_C = 1
        CONV1_OUT_C = 12
        CONV1_KERNEL = 8

        CONV2_OUT_C = 10
        CONV2_KERNEL = 8

        CONV3_OUT_C = 8
        CONV3_KERNEL = 6

        MP1_KERNEL = 3
        MP1_STRIDE = MP1_KERNEL

        FC1_IN_SIZE = 165*165*8
        FC1_OUT_SIZE = 128
        
        FC2_OUT_SIZE = 14
       
        # First 1x1 convolution before conv1 
        self.onebyone1 = nn.Conv2d(in_channels=CONV1_IN_C, out_channels=CONV1_IN_C, kernel_size=1)
        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.onebyone1.weight)

        # conv1: 1 input channel, 12 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=CONV1_IN_C, out_channels=CONV1_OUT_C, kernel_size=CONV1_KERNEL)
        
        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(CONV1_OUT_C)
        
        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)

        #TODO: Fill in the remaining initializations replacing each '_' with
        # the necessary value based on the provided specs for each layer

        # Second 1x1 convolution before conv2
        self.onebyone2 = nn.Conv2d(in_channels=CONV1_OUT_C, out_channels=CONV1_OUT_C, kernel_size=1)
        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.onebyone2.weight)

        #TODO: conv2: X input channels, 10 output channels, [8x8] kernel
        self.conv2 = nn.Conv2d(in_channels=CONV1_OUT_C, out_channels=CONV2_OUT_C, kernel_size=CONV2_KERNEL)
        self.conv2_normed = nn.BatchNorm2d(CONV2_OUT_C)
        torch_init.xavier_normal_(self.conv2.weight)

        # Third 1x1 convolution before conv3
        self.onebyone3 = nn.Conv2d(in_channels=CONV2_OUT_C, out_channels=CONV2_OUT_C, kernel_size=1)
        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.onebyone3.weight)

        #TODO: conv3: X input channels, 8 output channels, [6x6] kernel
        self.conv3 = nn.Conv2d(in_channels=CONV2_OUT_C, out_channels=CONV3_OUT_C, kernel_size=CONV3_KERNEL)
        self.conv3_normed = nn.BatchNorm2d(CONV3_OUT_C)
        torch_init.xavier_normal_(self.conv3.weight)

        #TODO: Apply max-pooling with a [3x3] kernel using tiling (*NO SLIDING WINDOW*)
        self.pool = nn.MaxPool2d(kernel_size=MP1_KERNEL, stride=MP1_STRIDE, padding=1)

        # Define 2 fully connected layers:
        #TODO: Use the value you computed in Part 1, Question 4 for fc1's in_features
        self.fc1 = nn.Linear(in_features=FC1_IN_SIZE, out_features=FC1_OUT_SIZE)
        self.fc1_normed = nn.BatchNorm1d(FC1_OUT_SIZE)
        torch_init.xavier_normal_(self.fc1.weight)

        #TODO: Output layer: what should out_features be?
        self.fc2 = nn.Linear(in_features=FC1_OUT_SIZE, out_features=FC2_OUT_SIZE).cuda()
        torch_init.xavier_normal_(self.fc2.weight)



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
        # First one by one
        batch = func.relu(self.onebyone1(batch))

        # Apply first convolution, followed by ReLU non-linearity; 
        # use batch-normalization on its outputs
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        
        # Second one by one
        batch = func.relu(self.onebyone2(batch))

        # Apply conv2 and conv3 similarly
        batch = func.relu(self.conv2_normed(self.conv2(batch)))

        # Third one by one
        batch = func.relu(self.onebyone3(batch))

        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        
        
        # Pass the output of conv3 to the pooling layer
        batch = self.pool(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))
        

        # Connect the reshaped features of the pooled conv3 to fc1
        batch = func.relu(self.fc1(batch))
        
        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        batch = self.fc2(batch)


        # Return the class predictions
        #TODO: apply an activition function to 'batch'
        return func.sigmoid(batch)
    
    

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
    model = BasicCNN()
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    criterion = nn.BCEWithLogitsLoss(weight=None) #TODO - loss criteria are defined in the torch.nn package

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
                v_acc = (tp+tn)/(tp+tn+fp+fn)

                val_loss.append(loss.item())
                val_acc.append(v_acc)

                if (loss < best_loss):
                    torch.save(model.state_dict(), 'best_baseline_model.pt')
                    best_loss = loss.item()
                 
                print('Epoch %d, average minibatch %d loss: %.3f, average acc: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss, n_train_acc))
                print('Epoch %d, validation loss: %f, validation acc: %.3f' % (epoch+1, loss.item(), v_acc))
                print('TP: %d, TN: %d, FN: %d, FP: %d' % (tp,tn,fn,fp))
                #print('Precision %f, Recall %f' % (tp,fn))
                
                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0
                n_train_acc = 0.0

        print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")

    train_data = np.array([avg_minibatch_loss, avg_train_acc, val_loss, val_acc]) 
    np.save('baseline_data.npy', train_data)

if __name__ == '__main__':
    main()
