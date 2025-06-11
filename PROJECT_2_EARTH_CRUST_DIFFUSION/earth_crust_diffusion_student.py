"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1       # 热扩散率 (m²/day)
A = 10.0      # 年平均地表温度 (°C)
B = 12.0      # 地表温度振幅 (°C)
TAU = 365.0   # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)

def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366, years=10):
    """
    求解地壳热扩散方程 (显式差分格式)
    
    参数:
        h (float): 空间步长 (m)
        a (float): 时间步长比例因子
        M (int): 深度方向网格点数
        N (int): 时间步数
        years (int): 总模拟年数
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - temperature_matrix (ndarray): 温度矩阵 [time, depth]
    """
    # 计算稳定性参数（注意公式与参考答案一致：r = (D * dt) / (dz²)，其中 dt = a²/(h*D)）
    dt = a ** 2 / (h * D)  # 由a=√(D*dt*M)推导而来，确保r=1
    r = D * dt / (h ** 2)  
    print(f"稳定性参数 r = {r:.4f}, 时间步长 dt = {dt:.2f} 天")

    # 初始化温度矩阵 [深度, 时间]（注意维度顺序与参考答案一致）
    T = np.zeros((M, N)) + T_INITIAL
    T[-1, :] = T_BOTTOM  # 底部边界条件（深度最大处为最后一行）

    # 时间步进循环（按年循环，总时间为years*N天）
    for _ in range(years):
        for j in range(N-1):  # 遍历每个时间步（从0到N-2）
            # 应用地表边界条件（正弦周期变化）
            T[0, j] = A + B * np.sin(2 * np.pi * j / TAU)
            
            # 显式差分格式（内部节点更新，排除边界）
            T[1:-1, j+1] = T[1:-1, j] + r * (
                T[2:, j] + T[:-2, j] - 2 * T[1:-1, j]
            )

    # 创建深度数组（注意深度方向从0到DEPTH_MAX，共M个点）
    depth = np.linspace(0, DEPTH_MAX, M)
    return depth, T.T  # 转置为[时间, 深度]维度顺序以符合常规习惯

def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365]):
    """
    绘制季节性温度轮廓（深度向下为正）
    """
    plt.figure(figsize=(8, 6))
    
    for day in seasons:
        # 取最后一年的对应时间点（避免初始年份过渡影响）
        idx = -1 * N + day  # 计算多年模拟中最后一年的索引
        plt.plot(temperature[idx, :], depth, 
                 label=f'Day {day}', linewidth=2, marker='o')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title('Seasonal Temperature Profiles in Earth Crust')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.gca().invert_yaxis()  # 深度向下为正
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行模拟（使用默认参数）
    depth, T = solve_earth_crust_diffusion()
    print(f"温度场形状: {T.shape}，深度点数: {len(depth)}")
    
    # 绘制典型季节的温度剖面（默认四季：春分、夏至、秋分、冬至）
    plot_seasonal_profiles(depth, T)
