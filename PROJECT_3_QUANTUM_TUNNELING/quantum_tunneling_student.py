"""学生模板：量子隧穿效应
文件：quantum_tunneling_student.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import linalg

class QuantumTunnelingSolver:
    """量子隧穿求解器类
    
    该类实现了一维含时薛定谔方程的数值求解，用于模拟量子粒子的隧穿效应。
    使用变形的Crank-Nicolson方法进行时间演化，确保数值稳定性和概率守恒。
    """
    
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """初始化量子隧穿求解器
        
        参数:
            Nx (int): 空间网格点数，默认220
            Nt (int): 时间步数，默认300
            x0 (float): 初始波包中心位置，默认40
            k0 (float): 初始波包动量(波数)，默认0.5
            d (float): 初始波包宽度参数，默认10
            barrier_width (int): 势垒宽度，默认3
            barrier_height (float): 势垒高度，默认1.0
        """
        # 初始化所有参数
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = int(barrier_width)  # 确保是整数
        self.barrier_height = barrier_height
        
        # 创建空间网格
        self.x = np.arange(self.Nx)
        
        # 设置势垒
        self.V = self.setup_potential()
        
        # 初始化波函数矩阵和系数矩阵
        self.C = np.zeros((self.Nx, self.Nt), dtype=complex)
        self.B = np.zeros((self.Nx, self.Nt), dtype=complex)
        
        # 构建系数矩阵
        self.A = self.build_coefficient_matrix()

    def wavefun(self, x):
        """高斯波包函数
        
        参数:
            x (np.ndarray): 空间坐标数组
            
        返回:
            np.ndarray: 初始波函数值
            
        数学公式:
            ψ(x,0) = exp(ik₀x) * exp(-(x-x₀)²ln10(2)/d²)
            
        物理意义:
            描述一个在x₀位置、具有动量k₀、宽度为d的高斯波包
        """
        # 实现高斯波包函数
        # 动量项
        momentum_term = np.exp(1j * self.k0 * x)
        # 高斯包络
        gaussian_term = np.exp(-(x - self.x0)**2 * np.log(10) * 2 / self.d**2)
        return momentum_term * gaussian_term

    def setup_potential(self):
        """设置势垒函数
        
        返回:
            np.ndarray: 势垒数组
            
        说明:
            在空间网格中间位置创建矩形势垒
            势垒位置：从 Nx//2 到 Nx//2+barrier_width
            势垒高度：barrier_height
        """
        # 创建势垒数组
        V = np.zeros(self.Nx)
        # 在中间位置设置势垒
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        V[barrier_start:barrier_end] = self.barrier_height
        return V

    def build_coefficient_matrix(self):
        """构建变形的Crank-Nicolson格式的系数矩阵
        
        返回:
            np.ndarray: 系数矩阵A
            
        数学原理:
            对于dt=1, dx=1的情况，哈密顿矩阵的对角元素为: -2+2j-V
            非对角元素为1（表示动能项的有限差分）
            
        矩阵结构:
            三对角矩阵，主对角线为 -2+2j-V[i]，上下对角线为1
        """
        # 构建系数矩阵
        # 主对角线
        main_diag = -2 + 2j - self.V
        # 上对角线和下对角线
        off_diag = np.ones(self.Nx-1)
        
        # 创建三对角矩阵
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    def solve_schrodinger(self):
        """求解一维含时薛定谔方程
        
        使用Crank-Nicolson方法进行时间演化
        
        返回:
            tuple: (x, V, B, C) - 空间网格, 势垒, 波函数矩阵, chi矩阵
            
        数值方法:
            Crank-Nicolson隐式格式，具有二阶精度和无条件稳定性
            时间演化公式：C[:,t+1] = 4j * solve(A, B[:,t])
                         B[:,t+1] = C[:,t+1] - B[:,t]
        """
        # 构建系数矩阵A
        A = self.A
        
        # 设置初始波函数
        self.B[:, 0] = self.wavefun(self.x)
        
        # 对初始波函数进行归一化
        norm = np.sum(np.abs(self.B[:, 0])**2)
        self.B[:, 0] /= np.sqrt(norm)
        
        # 时间循环：使用线性方程组求解进行时间演化
        for t in range(self.Nt-1):
            # 求解线性方程组 A·C[:,t+1] = 4j·B[:,t]
            self.C[:, t+1] = 4j * linalg.solve(A, self.B[:, t])
            
            # 更新B
            self.B[:, t+1] = self.C[:, t+1] - self.B[:, t]
        
        return self.x, self.V, self.B, self.C

    def calculate_coefficients(self):
        """计算透射和反射系数
        
        返回:
            tuple: (T, R) - 透射系数和反射系数
            
        物理意义:
            透射系数T：粒子穿过势垒的概率
            反射系数R：粒子被势垒反射的概率
            应满足：T + R ≈ 1（概率守恒）
            
        计算方法:
            T = ∫|ψ(x>barrier)|²dx / ∫|ψ(x)|²dx
            R = ∫|ψ(x<barrier)|²dx / ∫|ψ(x)|²dx
        """
        # 确定势垒位置
        barrier_end = self.Nx // 2 + self.barrier_width
        
        # 计算最后时刻的波函数概率密度
        prob_density = np.abs(self.B[:, -1])**2
        
        # 计算透射区域的概率（势垒右侧）
        T = np.sum(prob_density[barrier_end:])
        
        # 计算反射区域的概率（势垒左侧）
        R = np.sum(prob_density[:self.Nx//2])
        
        # 归一化处理
        total_prob = T + R
        T /= total_prob
        R /= total_prob
        
        return T, R

    def plot_evolution(self, time_indices=None):
        """绘制波函数演化图
        
        参数:
            time_indices (list): 要绘制的时间索引列表，默认为[0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
            
        功能:
            在多个子图中显示不同时刻的波函数概率密度和势垒
        """
        # 设置默认时间索引
        if time_indices is None:
            time_indices = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        
        # 创建子图布局
        fig, axes = plt.subplots(len(time_indices), 1, figsize=(10, 12), sharex=True)
        
        # 确保axes是可迭代的
        if len(time_indices) == 1:
            axes = [axes]
        
        # 对每个时间点绘图
        for i, t_idx in enumerate(time_indices):
            # 计算概率密度
            prob_density = np.abs(self.B[:, t_idx])**2
            
            # 绘制概率密度
            axes[i].plot(self.x, prob_density, 'b-', label=f't = {t_idx}')
            
            # 绘制势垒（缩放以显示在图中）
            scaled_V = self.V * 0.5 * np.max(prob_density) / self.barrier_height
            axes[i].plot(self.x, scaled_V, 'r-', label='Potential')
            
            # 添加标题和标签
            axes[i].set_title(f'Wavefunction at t = {t_idx}')
            axes[i].set_ylabel('Probability Density')
            axes[i].legend()
            axes[i].grid(True)
        
        # 设置x轴标签
        axes[-1].set_xlabel('Position')
        
        plt.tight_layout()
        return fig

    def create_animation(self, interval=20):
        """创建波包演化动画
        
        参数:
            interval (int): 动画帧间隔(毫秒)，默认20
            
        返回:
            matplotlib.animation.FuncAnimation: 动画对象
            
        功能:
            实时显示波包在势垒附近的演化过程
        """
        # 设置图形和坐标轴
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制势垒（缩放以显示在图中）
        max_prob = np.max(np.abs(self.B)**2)
        scaled_V = self.V * 0.5 * max_prob / self.barrier_height
        ax.plot(self.x, scaled_V, 'r-', label='Potential')
        
        # 创建线条对象
        line, = ax.plot([], [], 'b-', lw=2, label='Probability Density')
        
        # 设置坐标轴范围
        ax.set_xlim(0, self.Nx)
        ax.set_ylim(0, 1.2 * max_prob)
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability Density')
        ax.set_title('Quantum Tunneling: Wave Packet Evolution')
        ax.legend()
        ax.grid(True)
        
        # 定义动画更新函数
        def update(frame):
            line.set_data(self.x, np.abs(self.B[:, frame])**2)
            return line,
        
        # 使用 FuncAnimation 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=range(0, self.Nt, max(1, self.Nt//100)),
            interval=interval, blit=True
        )
        
        return ani

    def verify_probability_conservation(self):
        """验证概率守恒
        
        返回:
            np.ndarray: 每个时间步的总概率
            
        物理原理:
            量子力学中概率必须守恒：∫|ψ(x,t)|²dx = 常数
            数值计算中应该保持在1附近
        """
        # 计算每个时间步的总概率
        probabilities = np.zeros(self.Nt)
        for t in range(self.Nt):
            probabilities[t] = np.sum(np.abs(self.B[:, t])**2)
        
        return probabilities

    def demonstrate(self):
        """演示量子隧穿效应
        
        功能:
            1. 求解薛定谔方程
            2. 计算并显示透射和反射系数
            3. 绘制波函数演化图
            4. 验证概率守恒
            5. 创建并显示动画
            
        返回:
            animation对象
        """
        # 打印开始信息
        print("开始量子隧穿效应模拟...")
        print(f"势垒宽度: {self.barrier_width}")
        print(f"势垒高度: {self.barrier_height}")
        
        # 求解薛定谔方程
        self.solve_schrodinger()
        
        # 计算并显示系数
        T, R = self.calculate_coefficients()
        print(f"透射系数 T = {T:.4f}")
        print(f"反射系数 R = {R:.4f}")
        print(f"T + R = {T + R:.4f}")
        
        # 绘制演化图
        self.plot_evolution()
        
        # 验证概率守恒
        probabilities = self.verify_probability_conservation()
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.Nt), probabilities)
        plt.title('Probability Conservation')
        plt.xlabel('Time Step')
        plt.ylabel('Total Probability')
        plt.grid(True)
        
        # 创建动画
        ani = self.create_animation()
        
        return ani


def demonstrate_quantum_tunneling():
    """便捷的演示函数
    
    创建默认参数的求解器并运行演示
    
    返回:
        animation对象
    """
    # 创建求解器实例并调用demonstrate方法
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
    
    # 显示动画
    plt.show()
