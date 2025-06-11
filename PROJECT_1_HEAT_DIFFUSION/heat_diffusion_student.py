"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置初始条件：中间温度为100，两端为0
    u[int(Nx/2), 0] = 100
    
    # 计算扩散系数
    alpha = D * dt / dx**2
    
    # 确保满足稳定性条件
    if alpha > 0.5:
        print(f"警告: 数值解可能不稳定，alpha = {alpha} > 0.5")
    
    # 时间迭代
    for t in range(0, Nt-1):
        # 空间迭代
        for x in range(1, Nx-1):
            # 显式差分格式
            u[x, t+1] = u[x, t] + alpha * (u[x+1, t] - 2*u[x, t] + u[x-1, t])
    
    return u

def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    u = np.zeros((Nx, Nt))
    x = np.linspace(0, L, Nx)
    
    for t in range(Nt):
        time = t * dt
        for i, xi in enumerate(x):
            # 初始条件：中间温度为100，两端为0
            if i == int(Nx/2):
                u[i, t] = 100
            else:
                # 傅里叶级数解析解
                for n in range(1, n_terms+1):
                    An = (200 / (n * np.pi)) * np.sin(n * np.pi / 2)
                    u[i, t] += An * np.sin(n * np.pi * xi / L) * np.exp(-(n * np.pi / L)**2 * D * time)
    
    return u

def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    # 测试不同的时间步长
    dt_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    stability_results = []
    
    for dt_val in dt_values:
        alpha = D * dt_val / dx**2
        stability_results.append(alpha)
        
        print(f"时间步长 dt = {dt_val}s, alpha = {alpha}")
        if alpha > 0.5:
            print("  警告: 数值解可能不稳定")
        else:
            print("  数值解应该稳定")
    
    # 绘制稳定性分析图
    plt.figure(figsize=(10, 6))
    plt.plot(dt_values, stability_results, 'o-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='稳定性阈值')
    plt.xlabel('时间步长 dt (s)')
    plt.ylabel('扩散系数 alpha')
    plt.title('数值解稳定性分析')
    plt.legend()
    plt.grid(True)
    plt.savefig('stability_analysis.png')
    plt.show()

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置不同的初始条件：正弦分布
    x = np.linspace(0, L, Nx)
    u[:, 0] = 50 * np.sin(np.pi * x / L)
    
    # 计算扩散系数
    alpha = D * dt / dx**2
    
    # 时间迭代
    for t in range(0, Nt-1):
        # 空间迭代
        for x in range(1, Nx-1):
            # 显式差分格式
            u[x, t+1] = u[x, t] + alpha * (u[x+1, t] - 2*u[x, t] + u[x-1, t])
    
    return u

def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置初始条件：中间温度为100，两端为0
    u[int(Nx/2), 0] = 100
    
    # 计算扩散系数
    alpha = D * dt / dx**2
    
    # 冷却系数 (W/m^2/K)
    h = 10
    
    # 环境温度
    T_env = 25
    
    # 时间迭代
    for t in range(0, Nt-1):
        # 内部点的更新
        for x in range(1, Nx-1):
            u[x, t+1] = u[x, t] + alpha * (u[x+1, t] - 2*u[x, t] + u[x-1, t])
        
        # 边界条件：考虑冷却效应
        u[0, t+1] = u[0, t] + alpha * (u[1, t] - u[0, t]) - h * dt / (rho * C * dx) * (u[0, t] - T_env)
        u[Nx-1, t+1] = u[Nx-1, t] + alpha * (u[Nx-2, t] - u[Nx-1, t]) - h * dt / (rho * C * dx) * (u[Nx-1, t] - T_env)
    
    # 绘制温度随时间变化的曲线
    plt.figure(figsize=(12, 6))
    for t in [0, int(Nt/4), int(Nt/2), int(3*Nt/4), Nt-1]:
        plt.plot(np.linspace(0, L, Nx), u[:, t], label=f't = {t*dt:.1f}s')
    
    plt.xlabel('位置 (m)')
    plt.ylabel('温度 (°C)')
    plt.title('包含冷却效应的热传导')
    plt.legend()
    plt.grid(True)
    plt.savefig('heat_diffusion_with_cooling.png')
    plt.show()
    
    return u

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图
    
    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题
    
    返回:
        None
    
    示例:
        >>> u = np.zeros((100, 200))
        >>> plot_3d_solution(u, 0.01, 0.5, 200, "示例")
    """
    # 创建网格
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, dt*Nt, Nt)
    X, T = np.meshgrid(x, t)
    
    # 转置u以匹配meshgrid的形状
    U = u.T
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制表面图
    surf = ax.plot_surface(X, T, U, cmap='viridis', edgecolor='none')
    
    # 设置标签和标题
    ax.set_xlabel('位置 (m)')
    ax.set_ylabel('时间 (s)')
    ax.set_zlabel('温度 (°C)')
    ax.set_title(title)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 保存图像
    plt.savefig(f"{title.replace(' ', '_')}.png")
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    """
    主函数 - 演示和测试各任务功能
    
    执行顺序:
    1. 基本热传导模拟
    2. 解析解计算
    3. 数值解稳定性分析
    4. 不同初始条件模拟
    5. 包含冷却效应的热传导
    
    注意:
        学生需要先实现各任务函数才能正常运行
    """
    print("=== 铝棒热传导问题学生实现 ===")
    
    # 任务1: 基本热传导模拟
    print("\n任务1: 基本热传导模拟")
    u_basic = basic_heat_diffusion()
    plot_3d_solution(u_basic, dx, dt, Nt, "基本热传导模拟")
    
    # 任务2: 解析解计算
    print("\n任务2: 解析解计算")
    u_analytical = analytical_solution(n_terms=50)
    plot_3d_solution(u_analytical, dx, dt, Nt, "热传导解析解")
    
    # 任务3: 数值解稳定性分析
    print("\n任务3: 数值解稳定性分析")
    stability_analysis()
    
    # 任务4: 不同初始条件模拟
    print("\n任务4: 不同初始条件模拟")
    u_different = different_initial_condition()
    plot_3d_solution(u_different, dx, dt, Nt, "不同初始条件热传导")
    
    # 任务5: 包含冷却效应的热传导
    print("\n任务5: 包含冷却效应的热传导")
    u_cooling = heat_diffusion_with_cooling()
    plot_3d_solution(u_cooling, dx, dt, Nt, "包含冷却效应的热传导")
    
    print("\n所有任务完成!")
