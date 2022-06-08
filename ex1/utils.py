import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
import threading
import os


def plot_distance(ckpt, x, y, mode, p):
    distance_opt = []
    distance_real = []
    for i in range(len(ckpt)):
        distance_opt.append(np.linalg.norm(ckpt[i]-y, ord=2))
        distance_real.append(np.linalg.norm(ckpt[i]-x, ord=2))
    plt.figure()
    plt.grid()
    plt.title(f'{mode} p={p}')
    plt.plot(distance_real, label="distance to real")
    plt.plot(distance_opt, label="distance to opt")
    plt.legend()
    os.makedirs(f'./show/{mode}', exist_ok=True)
    plt.savefig(f'./show/{mode}/p_{p}.png', format='png')


def admmJob(client, z, lamb, x_new_list):
    """各个client单独进行训练"""
    x_new = client.ADMM_train(z, lamb)
    x_new_list.put(x_new)


def admmWork(clients, z, lamb_list, client_num):
    """控制client进行多线程训练"""
    x_new_list = []
    q_list = Queue()
    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=admmJob,
                             args=(clients[cid], z, lamb_list[cid], q_list))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    for _ in range(client_num):
        x_new_list.append(q_list.get())
    return x_new_list
