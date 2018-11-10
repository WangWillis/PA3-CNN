import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import os
from baseline_cnn import BasicCNN

# Setup: initialize the hyperparameters/variables
num_epochs = 1           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 1e-4 
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
    model = model.load_state_dict(torch.load('best_baseline_model.pt'))
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # GENERATE CONFUSION MATRIX:

    # Generate 14x14 confusion matrix
    confusion =np.zeros((14,14))

    # Begin training procedure
    for epoch in range(num_epochs):
        N = 50
        N_minibatch_loss = 0.0    
        n_train_acc = 0.0

        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)
            
            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            print(minibatch_count)
            print(outputs)
   

    # Divide each row by its row sum so that each row adds up to 1
    for row in range(confusion.shape[0]):
        confusion[row]=confusion[row]/sum(confusion[row])
    print(confusion)

if __name__ == '__main__':
    main()
