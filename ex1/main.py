from cProfile import label
from http import client
import numpy as np
from Server import *
from Client import *
from utils import *
import argparse


def main(args):
    np.random.seed(args.seed)
    client_num = 20
    mode = args.mode
    if mode == 'pg':
        epochs = 20000
        c = 0
        lr = 0.00001
        p = args.p
    elif mode == 'admm':
        epochs = 200
        c = 0.1
        lr = 1/c
        p = args.p
    elif mode == 'sg':
        epochs = 20000
        c = 0
        lr = 0.001
        p = args.p
    # 最优解
    x_opt = np.random.normal(loc=0, scale=1, size=[300])
    # 非零元素位置
    position = np.random.permutation(300)[:5]
    mask = np.zeros([300])
    mask[position] = 1
    # 得到稀疏度为5的向量
    x_opt = x_opt*mask

    # 初始解
    x = np.zeros(shape=[300])
    x_best = x
    lamb_list = np.zeros(shape=[client_num, 300])
    z = np.zeros(shape=[300])

    server = Server(lr, p, c)
    clients = [Client(x_opt, lr) for i in range(client_num)]

    ckpt = []
    print(
        f"Epoch: 0 Distance to opt: {np.linalg.norm(x_best-x_opt)}", flush=True)
    for epoch in range(1, epochs+1):
        if mode == 'pg':
            # 各个节点求梯度取平均
            grad_list = [clients[i].PG_train(x) for i in range(client_num)]
            # 邻近点投影
            x = server.PG_train(x, grad_list)
            ckpt.append(x)
            x_best = x
            if epoch % 1000 == 0:
                print(
                    f"Epoch: {epoch} Distance to opt: {np.linalg.norm(x_best-x_opt)}", flush=True)
        elif mode == 'admm':
            x_list = np.array(admmWork(clients, z, lamb_list, client_num))
            z, lamb_list = server.ADMM_train(x_list, lamb_list)
            ckpt.append(z)
            x_best = z
            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch} Distance to opt: {np.linalg.norm(x_best-x_opt)}", flush=True)
        elif mode == 'sg':
            # 递减步长
            cur_lr = lr/np.sqrt(epoch)
            # 各个节点求梯度取平均
            grad_list = [clients[i].PG_train(x) for i in range(client_num)]
            # 求解次梯度
            x = server.SG_train(x, grad_list, cur_lr)
            ckpt.append(x)
            x_best = x
            if epoch % 1000 == 0:
                print(
                    f"Epoch: {epoch} Distance to opt: {np.linalg.norm(x_best-x_opt)}", flush=True)

    plot_distance(ckpt, x_best, x_opt, mode, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='pg',
                        help='pg for proximal gradient, admm for ADMM, sg for stochastic gradient')
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--p", type=float, default=1)
    args = parser.parse_args()
    main(args)
