"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # 设置物理参数
    kappa = 1e-6  # 热扩散系数 (m^2/s)
    L = 100.0     # 模拟深度 (m)
    T_surface = 10.0  # 地表温度 (°C)
    T_bottom = 20.0   # 底部温度 (°C)
    A = 10.0        # 地表温度年变化幅度 (°C)
    P = 365.25 * 24 * 3600  # 周期 (秒)
    
    # 设置网格参数
    Nz = 101        # 深度方向网格点数
    dt = P/500      # 时间步长 (秒)
    Nt = 1000       # 时间步数
    z = np.linspace(0, L, Nz)  # 深度坐标数组 (m)
    dz = z[1] - z[0]  # 空间步长 (m)
    
    # 初始化温度场
    T = np.zeros((Nt, Nz))
    T[:, 0] = T_surface  # 地表温度边界条件
    T[:, -1] = T_bottom  # 底部温度边界条件
    
    # 计算稳定性条件
    alpha = kappa * dt / (dz**2)
    if alpha > 0.5:
        print(f"警告: 显式格式可能不稳定，当前 alpha = {alpha:.4f} > 0.5")
    
    # 实现显式差分格式
    for n in range(Nt-1):
        # 更新地表温度（周期性变化）
        T[n, 0] = T_surface + A * np.sin(2 * np.pi * n * dt / P)
        
        # 对内部节点应用显式差分格式
        for i in range(1, Nz-1):
            T[n+1, i] = T[n, i] + alpha * (T[n, i+1] - 2*T[n, i] + T[n, i-1])
    
    return z, T

if __name__ == "__main__":
    # 运行模拟
    depth, T = solve_earth_crust_diffusion()
    print(f"计算完成，温度场形状: {T.shape}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    # 绘制不同时间的温度剖面
    time_points = [0, T.shape[0]//4, T.shape[0]//2, 3*T.shape[0]//4, -1]
    colors = ['blue', 'green', 'red', 'purple', 'black']
    
    for i, t_idx in enumerate(time_points):
        plt.plot(T[t_idx, :], depth, color=colors[i], 
                 label=f't = {t_idx*solve_earth_crust_diffusion.__globals__["dt"]/86400:.1f} 天')
    
    plt.xlabel('温度 (°C)')
    plt.ylabel('深度 (m)')
    plt.title('地壳温度随深度和时间的变化')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # 深度向下为正
    plt.tight_layout()
    plt.show()
