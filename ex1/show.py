import matplotlib.pyplot as plt
import numpy as np
import os


def readFile(fileName):
    """读结果文件"""
    val_list = []
    with open(fileName) as f:
        for rows in f:
            val = rows.split(':')[-1]
            val = val[1:]
            val = float(val)
            val_list += [val]
    return val_list


def plot(pg_list, sg_list, admm_list, p_list):
    """画图"""
    plt.figure()
    plt.grid()
    plt.title(f'Proximal Gradient Descent')
    x = np.arange(0, pg_list.shape[1], 1)
    for i in range(len(pg_list)):
        plt.plot(1000*x,pg_list[i], label=f'p={p_list[i]}')
    plt.legend()
    plt.ylabel('Distance to Opt')
    plt.xlabel('Epoch')
    os.makedirs(f'./show', exist_ok=True)
    plt.savefig(f'./show/pg.png', format='png')

    plt.figure()
    plt.grid()
    plt.title(f'Subgradient Descent')
    for i in range(len(sg_list)):
        plt.plot(1000*x, sg_list[i], label=f'p={p_list[i]}')
    plt.legend()
    plt.ylabel('Distance to Opt')
    plt.xlabel('Epoch')
    os.makedirs(f'./show', exist_ok=True)
    plt.savefig(f'./show/sg.png', format='png')

    plt.figure()
    plt.grid()
    plt.title(f'ADMM')
    for i in range(len(admm_list)):
        plt.plot(10*x, admm_list[i], label=f'p={p_list[i]}')
    plt.legend()
    plt.ylabel('Distance to Opt')
    plt.xlabel('Epoch')
    os.makedirs(f'./show', exist_ok=True)
    plt.savefig(f'./show/admm.png', format='png')


if __name__ == '__main__':
    p_list = [0.01, 0.1, 1, 10, 100]
    pg_list = []
    sg_list = []
    admm_list = []
    for p in p_list:
        pg_list.append(readFile(f'result/pg/{p}.txt'))
        sg_list.append(readFile(f'result/sg/{p}.txt'))
        admm_list.append(readFile(f'result/admm/{p}.txt'))
    pg_list = np.array(pg_list)
    sg_list = np.array(sg_list)
    admm_list = np.array(admm_list)
    plot(pg_list, sg_list, admm_list, p_list)
