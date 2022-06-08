import numpy as np


class Server():
    def __init__(self, lr, p, c):
        """初始化参数"""
        self.p = p
        self.lr = lr
        self.c = c

    def soft_threshold(self, x_i, threshold):
        """软阈值更新"""
        if x_i < -threshold:
            x = x_i+threshold
        elif x_i > threshold:
            x = x_i-threshold
        else:
            x = 0
        return x

    def subgradient(self, x_i):
        """求解某个点x_i的次梯度"""
        if x_i == 0:
            return np.random.uniform(-1, 1)
        else:
            return np.sign(x_i)

    def PG_train(self, x, grad_list):
        """邻近点投影法"""
        grad_sum = np.sum(grad_list, axis=0)
        x_half = x - self.lr*grad_sum
        x = x_half
        for i in range(x_half.shape[0]):
            x[i] = self.soft_threshold(x[i], self.lr*self.p)
        return x

    def ADMM_train(self, x_list, lamb_list):
        """交替方向乘子法"""
        z = np.mean(x_list+lamb_list/self.c, axis=0)
        for i in range(z.shape[0]):
            z[i] = self.soft_threshold(z[i], self.p/self.c)
        for i in range(lamb_list.shape[0]):
            lamb_list[i] = lamb_list[i]+self.c*(x_list[i]-z)
        return z, lamb_list

    def SG_train(self, x, g_list, lr):
        """次梯度法"""
        l1_grad = np.zeros_like(x)
        for i in range(l1_grad.shape[0]):
            l1_grad[i] = self.subgradient(x[i])
        x = x-lr*(np.sum(g_list, axis=0)+self.p*l1_grad)
        return x
