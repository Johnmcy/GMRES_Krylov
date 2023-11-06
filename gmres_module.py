import numpy as np  # 导入numpy库，用于数值计算

# 定义GMRES算法函数
def gmres_function(A, b, x0, max_iterations, threshold, restart=None, M=None):
    if restart is None:  # 如果未设定重启次数，则将其设置为最大迭代次数
        restart = max_iterations

    if M is None:  # 如果未设定预处理矩阵，则将其设置为单位矩阵
        M = np.eye(A.shape[0])

    n = len(A)  # 获取系数矩阵A的维度
    m = restart  # 重启次数
    x = x0  # 设置初值
    r = b - np.dot(A, x)  # 计算初始残差
    b_norm = np.linalg.norm(b)  # 计算向量b的范数
    iterations = 0  # 初始化迭代次数
    numRestarts = 0  # 初始化重启计数
    e = []  # 初始化误差列表

    while iterations < max_iterations:  # 主迭代循环
        r = np.linalg.solve(M, r)  # 对残差应用预处理
        r_norm = np.linalg.norm(r)  # 计算残差范数
        Q = np.zeros((n, m + 1))  # 初始化Q矩阵
        Q[:, 0] = r / r_norm  # 第一列为单位化的残差
        beta = np.zeros(m + 1)  # 初始化beta向量
        beta[0] = r_norm  # 设置beta的第一个值
        cs = np.zeros(m)  # 初始化cosine值列表
        sn = np.zeros(m)  # 初始化sine值列表
        H = np.zeros((m + 1, m))  # 初始化Hessenberg矩阵

        # Arnoldi 过程
        for k in range(m):
            H[:k + 2, k], Q[:, k + 1] = arnoldi(A, Q, k, M)  # Arnoldi迭代
            H[:k + 2, k], cs[k], sn[k] = apply_givens_rotation(H[:k + 2, k], cs, sn, k)  # 应用Givens旋转
            beta[k + 1] = -sn[k] * beta[k]  # 更新beta值
            beta[k] = cs[k] * beta[k]  # 更新beta值
            error = abs(beta[k + 1]) / b_norm  # 计算误差
            e.append(error)  # 将误差值添加到列表
            if error <= threshold:  # 如果误差小于阈值，则退出循环
                break

        # 更新解
        y = np.linalg.solve(H[:k + 1, :k + 1], beta[:k + 1])  # 解小型线性系统以得到y
        x = x + np.dot(Q[:, :k + 1], y)  # 更新解

        r = b - np.dot(A, x)  # 计算新的残差
        iterations += k + 1  # 更新迭代次数
        if error <= threshold or iterations >= max_iterations:  # 检查终止条件
            break

        numRestarts += 1  # 更新重启计数

    # 检查是否使用了预处理
    preconditioned = M is not None and not np.array_equal(M, np.eye(A.shape[0]))
    return x, e, numRestarts, preconditioned  # 返回解、误差列表、重启次数和预处理标志

# 定义Arnoldi过程
def arnoldi(A, Q, k, M):
    q = np.dot(A, np.linalg.solve(M, Q[:, k]))  # 通过A和预处理矩阵M计算q
    h = np.zeros(k + 2)  # 初始化h向量
    for i in range(k + 1):  # 计算h的前k+1个元素
        h[i] = np.dot(q, Q[:, i])
        q = q - h[i] * Q[:, i]
    h[k + 1] = np.linalg.norm(q)  # 计算h的最后一个元素
    q = q / h[k + 1]  # 单位化q
    return h, q

# 定义应用Givens旋转的函数
def apply_givens_rotation(h, cs, sn, k):
    for i in range(k):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]  # 临时存储
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]  # 更新h[i+1]
        h[i] = temp  # 更新h[i]
    cs_k, sn_k = givens_rotation(h[k], h[k + 1])  # 计算新的cosine和sine值
    h[k] = cs_k * h[k] + sn_k * h[k + 1]  # 更新h[k]
    h[k + 1] = 0.0  # 将h[k+1]设为0
    return h, cs_k, sn_k

# 定义计算Givens旋转的cosine和sine值的函数
def givens_rotation(v1, v2):
    t = np.sqrt(v1 ** 2 + v2 ** 2)  # 计算t值
    cs = v1 / t  # 计算cosine值
    sn = v2 / t  # 计算sine值
    return cs, sn  # 返回cosine和sine值
