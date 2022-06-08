import matplotlib.pyplot as plt
import numpy as np
import os


def readFile(fileName):
    f1_list = []
    with open(fileName) as f:
        for rows in f:
            if(rows.startswith("---")):
                f1 = rows.split(':')[-1]
                f1 = f1[1:-8]
                f1 = float(f1)
                f1_list += [f1]
    return np.array(f1_list)


def plot(gd, sag, sgd_8, sgd_16, sgd_32, sgd_64, sgd_128):
    plt.figure()
    plt.grid()
    plt.title(f'F1 Curve')
    plt.plot(gd, label="GD")
    plt.plot(sag, label="SAG")
    plt.plot(sgd_16, label="SGD batch_size=32", color='green')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xlim([0,100])
    os.makedirs(f'./show', exist_ok=True)
    plt.savefig(f'./show/optimizer.png', format='png')

    plt.figure()
    plt.grid()
    plt.title(f'F1 Curve')
    plt.xlim([0,100])
    plt.plot(sgd_8, label="SGD batch_size=8")
    plt.plot(sgd_16, label="SGD batch_size=16")
    plt.plot(sgd_32, label="SGD batch_size=32", color='green')
    plt.plot(sgd_64, label="SGD batch_size=64")
    plt.plot(sgd_128, label="SGD batch_size=128")
    plt.ylim([0.9, 1])
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    os.makedirs(f'./show', exist_ok=True)
    plt.savefig(f'./show/SGD_batch_size.png', format='png')


if __name__ == '__main__':
    gd = readFile('GD.txt')
    sag = readFile('SAG.txt')
    sgd_8 = readFile('SGD_8.txt')
    sgd_16 = readFile('SGD_16.txt')
    sgd_32 = readFile('SGD_32.txt')
    sgd_64 = readFile('SGD_64.txt')
    sgd_128 = readFile('SGD_128.txt')
    plot(gd, sag, sgd_8, sgd_16, sgd_32, sgd_64, sgd_128)
