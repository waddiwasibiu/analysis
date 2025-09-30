import numpy as np

# 定义节点、单元和位移场
nodes = np.array([[0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 1]])

elements = np.array([[0, 1, 3],
                     [1, 2, 3]])

# 创建参考点M（参考三角形的中心）
Mref = np.array([1/3, 1/3])

# 三角形P1单元的形函数
def Nref_triangle_P1(xref):
    Nref = np.array([1-xref[0]-xref[1],
                     xref[0],
                     xref[1]])
    return Nref

# 测试形函数
print("形函数测试:")
print(Nref_triangle_P1(np.array([0,0])))
print(Nref_triangle_P1(np.array([1,0])))
print(Nref_triangle_P1(np.array([0,1])))
print(Nref_triangle_P1(Mref))

# 保存TP2的输出
np.savetxt("output_2.txt", Nref_triangle_P1(Mref).reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 位移插值函数
def interpolate_displacement(nodes, K, displacements, Nref):
    nb_nodes_loc = K.shape[0]
    dimension = nodes.shape[1]
    u = np.zeros(dimension)
    for i in range(dimension):
        for nloc in range(nb_nodes_loc):
            n = K[nloc]
            u[i] += Nref[nloc] * displacements[n][i]
    return u

# 定义位移场并测试插值函数
displacements = np.array([[0, 0],
                          [0.5, 0],
                          [1, 1],
                          [0.5, 1]])

u0_M = interpolate_displacement(nodes, elements[0,:], displacements, Nref_triangle_P1(Mref))
print("\n位移插值结果:")
print(u0_M)
u1_M = interpolate_displacement(nodes, elements[1,:], displacements, Nref_triangle_P1(Mref))
print(u1_M)

# 保存TP3的输出
np.savetxt("output_3.txt", np.array([u0_M,u1_M]).reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 三角形P1单元形函数的梯度
def gradNref_triangle_P1(xref):
    # P1单元的梯度是常数，与位置无关
    gradNref = np.array([[-1, -1],  # dN1/dξ, dN1/dη
                         [1, 0],    # dN2/dξ, dN2/dη
                         [0, 1]])   # dN3/dξ, dN3/dη
    return gradNref

# 测试梯度函数
print("\n形函数梯度测试:")
print(gradNref_triangle_P1(Mref))

# 保存TP4的输出
np.savetxt("output_4.txt", gradNref_triangle_P1(Mref).reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 计算变换F的梯度
def compute_gradF(nodes, K, gradNref):
    nb_nodes_loc = K.shape[0]
    dimension = nodes.shape[1]
    gradF = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            for nloc in range(nb_nodes_loc):
                n = K[nloc]
                gradF[i, j] += nodes[n, i] * gradNref[nloc, j]
    return gradF

# 测试梯度F计算
gradF0_M = compute_gradF(nodes, elements[0,:], gradNref_triangle_P1(Mref))
print("\n梯度F计算结果:")
print(gradF0_M)
gradF1_M = compute_gradF(nodes, elements[1,:], gradNref_triangle_P1(Mref))
print(gradF1_M)

# 保存TP5的输出
np.savetxt("output_5.txt", np.array([gradF0_M,gradF1_M]).reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 计算物理坐标系下的形函数梯度
def compute_gradN(K, gradF, gradNref):
    nb_nodes_loc = K.shape[0]
    dimension = gradF.shape[0]
    inv_gradF = np.linalg.inv(gradF)  # 计算梯度F的逆
    gradN = np.zeros((nb_nodes_loc, dimension))
    
    for nloc in range(nb_nodes_loc):
        for i in range(dimension):
            for j in range(dimension):
                gradN[nloc, i] += inv_gradF[i, j] * gradNref[nloc, j]
    return gradN

# 测试物理坐标系下的形函数梯度
gradN0_M = compute_gradN(elements[0,:], gradF0_M, gradNref_triangle_P1(Mref))
print("\n物理坐标系下的形函数梯度:")
print(gradN0_M)
gradN1_M = compute_gradN(elements[1,:], gradF1_M, gradNref_triangle_P1(Mref))
print(gradN1_M)

# 保存TP6的输出
np.savetxt("output_6.txt", np.array([gradN0_M,gradN1_M]).reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 插值位移梯度
def interpolate_grad_displacement(nodes, K, displacements, gradN):
    nb_nodes_loc = K.shape[0]
    dimension = nodes.shape[1]
    gradu = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            for nloc in range(nb_nodes_loc):
                n = K[nloc]
                gradu[i, j] += displacements[n, i] * gradN[nloc, j]
    return gradu

# 测试位移梯度插值
gradu0_M = interpolate_grad_displacement(nodes, elements[0,:], displacements, gradN0_M)
print("\n位移梯度插值结果:")
print(gradu0_M)
gradu1_M = interpolate_grad_displacement(nodes, elements[1,:], displacements, gradN1_M)
print(gradu1_M)

# 保存TP7的输出
np.savetxt("output_7.txt", np.array([gradu0_M,gradu1_M]).reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 重新设置位移场和自由度
displacements = np.array([[0, 0],
                          [0, 0],
                          [0, 0.01],
                          [0, 0.01]])
dofs = np.array([[0, 0],  # 0表示固定自由度
                 [1, 0],  # 1表示自由自由度
                 [1, 0],
                 [0, 0]])

# 定义张量积和弹性张量C
def produit_tensoriel(u,v):
    res = np.zeros((u.shape[0],v.shape[0]))
    for i in range(u.shape[0]):
        for j in range(v.shape[0]):
            res[i,j] = u[i]*v[j]
    return res

I2 = np.eye(2)
I4s = np.zeros((2,2,2,2))
for i in range(2):
    for j in range(2):
       for k in range(2):
            for l in range(2):
                I4s[i,j,k,l] = 0.5*((i == k) * (j == l) + (i == l) * (j == k))

_mu = 4
_lambda = 2
C_elastic_s = 2*_mu*I4s + _lambda*produit_tensoriel(I2.reshape(2*2),I2.reshape(2*2)).reshape(2,2,2,2)
print("\n弹性张量C:")
print(C_elastic_s.reshape(2*2,2*2))

# 计算局部刚度矩阵和组装全局矩阵
dimension = nodes.shape[1]
nb_nodes = nodes.shape[0]
nb_elements = elements.shape[0]
nb_nodes_per_element = elements.shape[1]

# 初始化局部刚度矩阵和载荷向量
matKlocs = np.zeros((nb_elements, nb_nodes_per_element, dimension, 
                    nb_nodes_per_element, dimension))
vecBlocs = np.zeros((nb_elements, nb_nodes_per_element, dimension))

# 初始化全局刚度矩阵和载荷向量
matK = np.zeros((nb_nodes, dimension, nb_nodes, dimension))
vecB = np.zeros((nb_nodes, dimension))

for id_element in range(nb_elements):
    K = elements[id_element,:]
    print(f"\n单元 {K} 的贡献:")
    
    # 高斯积分点（重心）
    xref = np.array([1/3, 1/3])
    wref = 0.5  # 三角形单元的权重
    
    # 计算形函数梯度和变换梯度
    gradNref_q = gradNref_triangle_P1(xref)
    gradF_q = compute_gradF(nodes, K, gradNref_q)
    gradN_q = compute_gradN(K, gradF_q, gradNref_q)
    detF_q = np.linalg.det(gradF_q)
    
    # 初始化局部刚度矩阵
    matKloc = np.zeros((nb_nodes_per_element, dimension, 
                       nb_nodes_per_element, dimension))
    vecBloc = np.zeros((nb_nodes_per_element, dimension))
    
    # 计算局部刚度矩阵
    for mloc in range(nb_nodes_per_element):
        for i in range(dimension):
            for nloc in range(nb_nodes_per_element):
                for k in range(dimension):
                    for j in range(dimension):
                        for l in range(dimension):
                            matKloc[mloc, i, nloc, k] += (
                                C_elastic_s[i, j, k, l] * gradN_q[mloc, j] * 
                                gradN_q[nloc, l] * detF_q * wref
                            )
    
    print("局部刚度矩阵:")
    print(matKloc.reshape(nb_nodes_per_element*dimension, nb_nodes_per_element*dimension))
    print("局部载荷向量:")
    print(vecBloc.reshape(nb_nodes_per_element*dimension, 1))
    
    matKlocs[id_element] = matKloc
    vecBlocs[id_element] = vecBloc
    
    # 组装到全局矩阵
    for mloc in range(nb_nodes_per_element):
        m = K[mloc]
        for i in range(dimension):
            for nloc in range(nb_nodes_per_element):
                n = K[nloc]
                for k in range(dimension):
                    matK[m, i, n, k] += matKloc[mloc, i, nloc, k]
            vecB[m, i] += vecBloc[mloc, i]

print("\n全局刚度矩阵:")
print(matK.reshape(nb_nodes*dimension, nb_nodes*dimension))
print("全局载荷向量:")
print(vecB.reshape(nb_nodes*dimension, 1))

# 保存TP8和TP9的输出
np.savetxt("output_8.txt", matKlocs.reshape(-1), '%.8e', delimiter='\n', newline='\n')
np.savetxt("output_9.txt", matK.reshape(-1), '%.8e', delimiter='\n', newline='\n')

# 处理Dirichlet边界条件
matKtilde = matK.copy().reshape(nb_nodes*dimension, nb_nodes*dimension)
vecBtilde = vecB.copy().reshape(nb_nodes*dimension, 1)

for m in range(nb_nodes):
    for i in range(dimension):
        if dofs[m, i] == 0:  # 固定自由度
            idx = m * dimension + i
            # 清零对应行
            matKtilde[idx, :] = 0
            # 设置对角元为1
            matKtilde[idx, idx] = 1
            # 设置载荷向量为固定位移值
            vecBtilde[idx] = displacements[m, i]

print("\n处理边界条件后的刚度矩阵:")
print(matKtilde)
print("处理边界条件后的载荷向量:")
print(vecBtilde)

# 保存TP10的输出
np.savetxt("output_10.txt", np.concatenate((matKtilde, vecBtilde), axis=1).reshape(-1), 
           '%.8e', delimiter='\n', newline='\n')

# 求解线性系统
displacements_sol = np.linalg.solve(matKtilde, vecBtilde).reshape(nb_nodes, dimension)
print("\n求解得到的位移:")
print(displacements_sol)

# 计算节点力
forces = np.dot(matK.reshape(nb_nodes*dimension, nb_nodes*dimension), 
                displacements_sol.reshape(nb_nodes*dimension, 1))
forces = forces.reshape(nb_nodes, dimension)
print("节点力:")
print(forces)

# 保存TP11的输出
np.savetxt("output_11.txt", forces.reshape(-1), '%.8e', delimiter='\n', newline='\n')
