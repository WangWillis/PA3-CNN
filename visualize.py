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
    model_weights = torch.load("./best_baseline_model.pt")
    np.save("baseline_conv1.npy", model_weights['conv1.weight'].cpu().numpy())
    np.save("baseline_conv2.npy", model_weights['conv2.weight'].cpu().numpy())
    np.save("baseline_conv3.npy", model_weights['conv3.weight'].cpu().numpy())

    conv1_weights = np.load("baseline_conv1.npy")
    conv2_weights = np.load("baseline_conv2.npy")
    conv3_weights = np.load("baseline_conv3.npy")
    fit =plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt in conv1_weights:
        plt.subplot(4,8,idx + 1)
        plt.imshow(filt[0,:,:], cmap="gray")
        plt.axis('off')
    fig.show()

if __name__ == '__main__':
    main()
