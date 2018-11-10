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
from res_model import ResCNN

def main():
    model = ResCNN()
    model.load_state_dict(torch.load("./best_res_model.pt"))
    conv1_weights = model.conv1.weight.detach().cpu().numpy()
    conv2_weights = model.layer1.blocks[0].conv1.weight.detach().cpu().numpy()
    conv3_weights = model.layer2.blocks[0].conv1.weight.detach().cpu().numpy()

    np.save('res_conv1_weights.npy', conv1_weights)
    np.save('res_conv2_weights.npy', conv2_weights)
    np.save('res_conv3_weights.npy', conv3_weights)
    
    print(conv1_weights.shape)
    print(conv2_weights.shape)
    print(conv3_weights.shape)
    plt.figure(figsize=(5,5))

    for i in range(conv1_weights.shape[0]):
        plt.subplot(8,4,i+1)
        plt.imshow(conv1_weights[i][0], cmap="gray")
        plt.axis('off')
    plt.savefig('res_conv1_vis.png')

    for i in range(conv2_weights.shape[0]):
        plt.subplot(8,4,i+1)
        plt.imshow(conv2_weights[i][0], cmap="gray")
        plt.axis('off')
    plt.savefig('res_conv2_vis.png')
    
    for i in range(conv3_weights.shape[0]):
        plt.subplot(8,8,i+1)
        plt.imshow(conv3_weights[i][0], cmap="gray")
        plt.axis('off')
    plt.savefig('res_conv3_vis.png')
if __name__ == '__main__':
    main()
