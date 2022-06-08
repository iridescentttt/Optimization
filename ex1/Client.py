import numpy as np


class Client():
    def __init__(self, x_opt, lr) -> None:
        """初始化随机变量"""
        self.e = np.random.normal(loc=0, scale=0.2, size=[10])
        self.A = np.random.normal(loc=0, scale=1, size=[10, 300])
        self.b = self.A@x_opt+self.e
        self.lr = lr

    def PG_train(self, x):
        """邻近点梯度法"""
        g = self.A.T @ (self.A @ x - self.b)
        return g

    def ADMM_train(self, z, lamb):
        """交替方向乘子法"""
        x_new = np.linalg.inv(self.A.T@self.A+self.lr *
                              np.eye(self.A.shape[1], self.A.shape[1]))@(self.A.T@self.b+self.lr*z-lamb)
        return x_new

    def distance(self, x):
        """计算距离"""
        return self.A @ x - self.b
