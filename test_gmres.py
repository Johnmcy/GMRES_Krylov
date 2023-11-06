import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
from gmres_module import gmres_function  # 从gmres_module模块导入gmres_function函数

# 定义测试GMRES的函数
def test_gmres():
    n = 50  # 设定系数矩阵的维度为50
    A = np.random.rand(n, n) + n * np.eye(n)  # 创建一个随机矩阵并加上一个对角矩阵以确保其正定性
    A = 0.5 * (A + A.T)  # 使矩阵A对称
    b = np.random.rand(n)  # 创建一个随机向量b
    x0 = np.zeros(n)  # 初始化解的初值为零向量
    max_iters = 100  # 设定最大迭代次数为100
    tol = 1e-6  # 设定误差容忍度为1e-6
    restart = 3  # 设定重启次数为3
    M = np.diag(1.0 / np.diag(A))  # 设定预处理矩阵为A的对角矩阵的逆

    # 调用GMRES函数求解线性系统
    x, e, numRestarts, preconditioned = gmres_function(A, b, x0, max_iters, tol, restart, M)

    # 计算解的残差
    residual = np.linalg.norm(np.dot(A, x) - b)

    # 显示结果
    print('Solution x:', x)  # 打印解x
    print('Error e:', e)  # 打印误差列表e
    print('Number of iterations:', len(e))  # 打印迭代次数
    print('Number of restarts:', numRestarts)  # 打印重启次数
    print('Preconditioned:', preconditioned)  # 打印是否使用了预处理
    print('Residual:', residual)  # 打印残差

    # 绘制误差随迭代次数的变化图
    plt.figure()  # 创建一个新的图形
    plt.semilogy(range(len(e)), e, '-o', linewidth=2)  # 以对数尺度绘制误差曲线
    plt.axhline(tol, color='r', linestyle='--', linewidth=2)  # 绘制误差容忍度的红色虚线
    plt.xlabel('Iterations')  # x轴标签为"Iterations"
    plt.ylabel('Error (log scale)')  # y轴标签为"Error (log scale)"
    plt.title('Error vs. Iterations (log scale)')  # 设置图标题
    plt.legend(['Error', 'Tolerance'])  # 设置图例
    plt.grid(True)  # 显示网格线
    plt.show()  # 显示图形

# 调用test_gmres函数进行测试
test_gmres()
