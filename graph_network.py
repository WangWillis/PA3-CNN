import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_data(train, val, x_name, y_name, train_label, val_label, filename, plotname, batch_offset=50):
    plt.title(plotname)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    mini_batches = []
    for i in range(len(train)):
        mini_batches.append((i+1)*batch_offset)

    plt.plot(mini_batches, train, label=train_label)
    plt.plot(mini_batches, val, label=val_label)

    plt.legend()
    plt.savefig(filename)
    plt.gcf().clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='.npy file of model data', required=True)

    args = parser.parse_args()

    model_data = np.load(args.filename)
    filename = args.filename[:-9]

    train_loss, train_acc, val_loss, val_acc = model_data
    
    plot_data(train_loss, val_loss, 'Minibatch', 'Loss', 'Training', 'Validation', filename+'_loss_graph.png', 'Training vs. Validation Loss')
    plot_data(train_acc, val_acc, 'Minibatch', 'Accuracy', 'Training', 'Validation', filename+'_acc_graph.png', 'Training vs. Validation Accuracy')
