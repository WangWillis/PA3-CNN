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

def main():
    model_weights = torch.load("./best_onebyone_model.pt")
    np.save("onebyone_conv1.npy", model_weights['conv1.weight'].cpu().numpy())
    np.save("onebyone_conv2.npy", model_weights['conv2.weight'].cpu().numpy())
    np.save("onebyone_conv3.npy", model_weights['conv3.weight'].cpu().numpy())

    conv1_weights = np.load("onebyone_conv1.npy")
    conv2_weights = np.load("onebyone_conv2.npy")
    conv3_weights = np.load("onebyone_conv3.npy")
    print(conv1_weights.shape)
    print(conv2_weights.shape)
    print(conv3_weights.shape)
    plt.figure(figsize=(5,5))

    for i in range(conv1_weights.shape[0]):
        plt.subplot(4,3,i+1)
        plt.imshow(conv1_weights[i][0], cmap="gray")
        plt.axis('off')
    plt.savefig('onebyone_conv1_vis.png')

    for i in range(conv2_weights.shape[0]):
        plt.subplot(5,2,i+1)
        plt.imshow(conv2_weights[i][0], cmap="gray")
        plt.axis('off')
    plt.savefig('onebyone_conv2_vis.png')
    
    for i in range(conv3_weights.shape[0]):
        plt.subplot(4,2,i+1)
        plt.imshow(conv3_weights[i][0], cmap="gray")
        plt.axis('off')
    plt.savefig('onebyone_conv3_vis.png')
if __name__ == '__main__':
    main()
