# 最优化实验
实验一
> 考虑一个 20 节点的分布式系统。节点 $\mathrm{i}$ 有线性测量 $b_{i}=A_{i} x+e_{i}$, 其中 $b_{i}$ 为 10 维的测量值, $A_{i}$ 为 $10 \times 300$ 维的测量矩阵, $x$ 为 300 维的末知稀疏向量且稀疏度为 5 , $e_{i}$ 为 10 维的测 量噪声。从所有 $b_{i}$ 与 $A_{i}$ 中恢复 $x$ 的一范数规范化最小二乘模型如下:
> $$
> \min (1 / 2)|| A_{1} x-b_{1}||_{2}{ }^{2}+\cdots+(1 / 2)|| A_{20} x-b_{20}||_{2}{ }^{2}+p|| x||_{1}
> $$
> 其中 $p$ 为非负的正则化参数。请设计下述分布式算法求解该问题:
> 1、邻近点梯度法;
> 2、交替方向乘子法;
> 3、次梯度法;
> 在实验中, 设 $\mathrm{x}$ 的真值中的非零元素服从均值为 0 方差为 1 的高斯分布, $\mathrm{A}_{\mathrm{i}}$ 中的元素服从均值为 0 方差为 1 的高斯分布, $e_{i}$ 中的元素服从均值为 0 方差为 $0.2$ 的高斯分布。对于每种算 法, 请给出每步计算结果与真值的距离以及每步计算结果与最优解的距离。此外, 请讨论正则化参数 $p$ 对计算结果的影响。

实验二
> 请设计下述算法，求解 MNIST 数据集上的分类问题：  
>
> 1、梯度下降法；  
>
> 2、随机梯度法；  
>
> 3、随机平均梯度法 SAG（选做）。  
>
> 对于每种算法，请给出每步计算结果在测试集上所对应的分类精度。对于随机梯度法，请讨论 mini-batch 大小的影响。可采用 Logistic Regression 模型或神经网络模型。

运行脚本为 run.sh.

结果在 show 文件夹下.