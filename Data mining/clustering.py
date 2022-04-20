import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
m=2

class FCM:
    def __init__(self, data, clust_num,iter_num=10):
        self.data = data
        self.cnum = clust_num
        self.sample_num=data.shape[0]
        self.dim = data.shape[-1]  # 数据最后一维度数
        Jlist=[]   # 存储目标函数计算值的矩阵

        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num): # 迭代次数默认为10
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            print("第%d次迭代" %(i+1) ,end="")
            print("聚类中心",C)
            J = self.J_calcu(self.data, U, C)  # 计算目标函数
            Jlist = np.append(Jlist, J)
        self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
        self.Clast = C    # 最后的类中心矩阵
        self.Jlist = Jlist  # 存储目标函数计算值的矩阵

    # 初始化隶属度矩阵U
    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)  # sample_num为样本个数, cluster_n为分类数
        row_sum = np.sum(U, axis=1)  # 按行求和 row_sum: sample_num*1
        row_sum = 1 / row_sum    # 该矩阵每个数取倒数
        U = np.multiply(U.T, row_sum)  # 确保U的每列和为1 (cluster_n*sample_num).*(sample_num*1)
        return U   # cluster_n*sample_num

    # 计算类中心
    def Cen_Iter(self, data, U, cluster_n):
        c_new = np.empty(shape=[0, self.dim])  # self.dim为样本矩阵的最后一维度
        for i in range(0, cluster_n):          # 如散点的dim为2，图片像素值的dim为1
            u_ij_m = U[i, :] ** m  # (sample_num,)
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)  # (dim,)
            ux = np.reshape(ux,(1, self.dim))  # (1,dim)
            c_new = np.append(c_new, ux / sum_u, axis=0)   # 按列的方向添加类中心到类中心矩阵
        return c_new  # cluster_num*dim

    # 隶属度矩阵迭代
    def U_Iter(self, U, c):
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /
                            np.linalg.norm(self.data[j, :] - c[k, :])) ** (
                                2 / (m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum

        return U

    # 计算目标函数值
    def J_calcu(self, data, U, c):
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** m

        J = np.sum(np.sum(temp1))
        print("目标函数值:%.2f" %J)
        return J


    # 打印聚类结果图
    def plot(self):

        mark = ['or', 'ob', 'og', 'om', 'oy', 'oc']  # 聚类点的颜色及形状

        if self.dim == 2:
            #第一张图
            plt.subplot(221)
            plt.plot(self.data[:, 0], self.data[:, 1],'ob',markersize=2)
            plt.title('未聚类前散点图')

            #第二张图
            plt.subplot(222)
            j = 0
            for i in self.label:
                plt.plot(self.data[j:j + 1, 0], self.data[j:j + 1, 1], mark[i],
                         markersize=2)
                j += 1

            plt.plot(self.Clast[:, 0], self.Clast[:, 1], 'k*', markersize=7)
            plt.title("聚类后结果")

            # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图",)

            plt.show()
        elif self.dim==1:

            plt.subplot(221)
            plt.title("聚类前散点图")
            for j in range(0, self.data.shape[0]):
                plt.plot(self.data[j, 0], 'ob',markersize=3)  # 打印散点图

            plt.subplot(222)
            j = 0
            for i in self.label:
                plt.plot(self.data[j:j + 1, 0], mark[i], markersize=3)
                j += 1

            plt.plot([0]*self.Clast.shape[0],self.Clast[:, 0], 'k*',label='聚类中心',zorder=2)
            plt.title("聚类后结果图")
            plt.legend()
            # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图", )
            plt.show()

        elif self.dim==3:
            # 第一张图
            fig = plt.figure()
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.scatter(self.data[:, 0], self.data[:, 1],self.data[:,2], "b")
            ax1.set_xlabel("X 轴")
            ax1.set_ylabel("Y 轴")
            ax1.set_zlabel("Z 轴")
            plt.title("未聚类前的图")

            # 第二张图
            ax2 = fig.add_subplot(222, projection='3d')

            j = 0

            for i in self.label:
                ax2.plot(self.data[j:j+1, 0], self.data[j:j+1, 1],self.data[j:j+1,2], mark[i],markersize=5)
                j += 1
            ax2.plot(self.Clast[:, 0], self.Clast[:, 1], self.Clast[:, 2], 'k*', label='聚类中心', markersize=8)

            plt.legend()

            ax2.set_xlabel("X 轴")
            ax2.set_ylabel("Y 轴")
            ax2.set_zlabel("Z 轴")
            plt.title("聚类后结果")
            # # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图", )
            plt.show()
def one():
    result1 = np.random.normal(-5, 2, [150,2])  # 均值为5,方差为2
    result2 = np.random.normal(0, 3, [150,2])  # 均值为5,方差为2
    result3 = np.random.normal(5, 1, [150,2])  # 均值为5,方差为2
    data2 =result1 + result2 + result3
    a = FCM(data2, 3, 20)
    a.plot()
def two():
    def dbmoon(N=200, d=2, r=10, w=2):
        N1 = 10 * N
        w2 = w / 2
        done = True
        data = np.empty(0)
        while done:
            # generate Rectangular data
            tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
            tmp_y = (r + w2) * np.random.random([N1, 1])
            tmp = np.concatenate((tmp_x, tmp_y), axis=1)
            tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
            # generate double moon data ---upper
            idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
            idx = (idx.nonzero())[0]

            if data.shape[0] == 0:
                data = tmp.take(idx, axis=0)
            else:
                data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
            if data.shape[0] >= N:
                done = False
        print(data)
        db_moon = data[0:N, :]
        print(db_moon)
        # generate double moon data ----down
        data_t = np.empty([N, 2])
        data_t[:, 0] = data[0:N, 0] + r
        data_t[:, 1] = -data[0:N, 1] - d
        db_moon = np.concatenate((db_moon, data_t), axis=0)
        return db_moon
    N = 200
    d = -2
    r = 10
    w = 2
    a = 0.1
    num_MSE = []
    num_step = []
    data = dbmoon(N, d, r, w)
    a=FCM(data,2,20)
    a.plot()
if __name__ == '__main__':
    one()
    two()
