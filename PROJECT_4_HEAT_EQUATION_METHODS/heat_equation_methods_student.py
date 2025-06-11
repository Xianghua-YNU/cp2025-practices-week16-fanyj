#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        # 创建零数组
        u = np.zeros(self.nx)
        
        # 设置初始条件（10 <= x <= 11 区域为1）
        mask = (self.x >= 10) & (self.x <= 11)
        u[mask] = 1.0
        
        # 应用边界条件（虽然已经是0，但为了明确）
        u[0] = 0
        u[-1] = 0
        
        return u
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 显式差分法直接从当前时刻计算下一时刻的解
        数值方法: 使用scipy.ndimage.laplace计算空间二阶导数
        稳定性条件: r = alpha * dt / dx² <= 0.5
        
        实现步骤:
        1. 检查稳定性条件
        2. 初始化解数组和时间
        3. 时间步进循环
        4. 使用laplace算子计算空间导数
        5. 更新解并应用边界条件
        6. 存储指定时间点的解
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算稳定性参数 r = alpha * dt / dx²
        r = self.alpha * dt / (self.dx ** 2)
        
        # 检查稳定性条件 r <= 0.5
        if r > 0.5:
            print(f"警告: 显式方法不稳定 (r = {r:.4f} > 0.5)")
        
        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0.0
        
        # 创建结果存储字典
        results = {'times': [], 'solutions': []}
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 记录开始时间
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_step = min(dt, self.T_final - t)
            r_step = self.alpha * dt_step / (self.dx ** 2)
            
            # 使用 laplace(u) 计算空间二阶导数
            u_laplacian = laplace(u) / (self.dx ** 2)
            
            # 更新解：u += r * laplace(u)
            u += r_step * u_laplacian
            
            # 应用边界条件
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_step
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < 1e-10 or (t > plot_time and abs(t - plot_time) < dt_step + 1e-10):
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
                    break
        
        # 记录结束时间
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 隐式差分法在下一时刻求解线性方程组
        数值方法: 构建三对角矩阵系统并求解
        优势: 无条件稳定，可以使用较大时间步长
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建三对角系数矩阵
        3. 时间步进循环
        4. 构建右端项
        5. 求解线性系统
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建三对角矩阵（内部节点）
        # 上对角线：-r
        # 主对角线：1 + 2r
        # 下对角线：-r
        n_internal = self.nx - 2
        diagonals = [
            -r * np.ones(n_internal - 1),  # 上对角线
            (1 + 2 * r) * np.ones(n_internal),  # 主对角线
            -r * np.ones(n_internal - 1)   # 下对角线
        ]
        offsets = [-1, 0, 1]
        A_banded = scipy.linalg.diags(diagonals, offsets).toarray()
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 记录开始时间
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_step = min(dt, self.T_final - t)
            r_step = self.alpha * dt_step / (self.dx ** 2)
            
            # 更新矩阵（如果时间步长变化）
            if abs(dt_step - dt) > 1e-10:
                diagonals = [
                    -r_step * np.ones(n_internal - 1),
                    (1 + 2 * r_step) * np.ones(n_internal),
                    -r_step * np.ones(n_internal - 1)
                ]
                A_banded = scipy.linalg.diags(diagonals, offsets).toarray()
            
            # 构建右端项（内部节点）
            rhs = u[1:-1].copy()
            
            # 使用 scipy.linalg.solve_banded 求解
            # 注意：solve_banded 期望的格式是 [上对角线, 主对角线, 下对角线]
            # 我们需要将矩阵转换为这种格式
            ab = np.zeros((3, n_internal))
            ab[0, 1:] = diagonals[0]  # 上对角线
            ab[1, :] = diagonals[1]   # 主对角线
            ab[2, :-1] = diagonals[2] # 下对角线
            
            u_internal = scipy.linalg.solve_banded((1, 1), ab, rhs)
            
            # 更新解并应用边界条件
            u[1:-1] = u_internal
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_step
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < 1e-10 or (t > plot_time and abs(t - plot_time) < dt_step + 1e-10):
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
                    break
        
        # 记录结束时间
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: Crank-Nicolson方法结合显式和隐式格式
        数值方法: 时间上二阶精度，无条件稳定
        优势: 高精度且稳定性好
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建左端矩阵 A
        3. 时间步进循环
        4. 构建右端向量
        5. 求解线性系统 A * u^{n+1} = rhs
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建左端矩阵 A（内部节点）
        # 上对角线：-r/2
        # 主对角线：1 + r
        # 下对角线：-r/2
        n_internal = self.nx - 2
        diagonals_A = [
            -(r/2) * np.ones(n_internal - 1),  # 上对角线
            (1 + r) * np.ones(n_internal),     # 主对角线
            -(r/2) * np.ones(n_internal - 1)   # 下对角线
        ]
        offsets = [-1, 0, 1]
        A_banded = scipy.linalg.diags(diagonals_A, offsets).toarray()
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 记录开始时间
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_step = min(dt, self.T_final - t)
            r_step = self.alpha * dt_step / (self.dx ** 2)
            
            # 更新矩阵（如果时间步长变化）
            if abs(dt_step - dt) > 1e-10:
                diagonals_A = [
                    -(r_step/2) * np.ones(n_internal - 1),
                    (1 + r_step) * np.ones(n_internal),
                    -(r_step/2) * np.ones(n_internal - 1)
                ]
                A_banded = scipy.linalg.diags(diagonals_A, offsets).toarray()
            
            # 构建右端向量：(r/2)*u[:-2] + (1-r)*u[1:-1] + (r/2)*u[2:]
            rhs = (r_step/2) * u[:-2] + (1 - r_step) * u[1:-1] + (r_step/2) * u[2:]
            
            # 考虑边界条件对右端项的影响
            # 由于u[0]和u[-1]都是0，所以不需要额外处理
            
            # 求解线性系统
            # 注意：solve_banded 期望的格式是 [上对角线, 主对角线, 下对角线]
            ab = np.zeros((3, n_internal))
            ab[0, 1:] = diagonals_A[0]  # 上对角线
            ab[1, :] = diagonals_A[1]   # 主对角线
            ab[2, :-1] = diagonals_A[2] # 下对角线
            
            u_internal = scipy.linalg.solve_banded((1, 1), ab, rhs)
            
            # 更新解并应用边界条件
            u[1:-1] = u_internal
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_step
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < 1e-10 or (t > plot_time and abs(t - plot_time) < dt_step + 1e-10):
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
                    break
        
        # 记录结束时间
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度
            
        返回:
            np.ndarray: 内部节点的时间导数
            
        物理背景: 将PDE转化为ODE系统
        数值方法: 使用laplace算子计算空间导数
        
        实现步骤:
        1. 重构包含边界条件的完整解
        2. 使用laplace计算二阶导数
        3. 返回内部节点的导数
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 使用 laplace(u_full) / dx² 计算二阶导数
        u_laplacian = laplace(u_full) / (self.dx ** 2)
        
        # 返回内部节点的时间导数：alpha * d²u/dx²
        return self.alpha * u_laplacian[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        参数:
            method (str): 积分方法（'RK45', 'BDF', 'Radau'等）
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 将PDE转化为ODE系统求解
        数值方法: 使用高精度ODE求解器
        优势: 自适应步长，高精度
        
        实现步骤:
        1. 提取内部节点初始条件
        2. 调用solve_ivp求解ODE系统
        3. 重构包含边界条件的完整解
        4. 返回结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 提取内部节点初始条件
        u0_internal = self.u_initial[1:-1]
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用 solve_ivp 求解
        # fun: self._heat_equation_ode
        # t_span: (0, T_final)
        # y0: 内部节点初始条件
        # method: 指定的积分方法
        # t_eval: plot_times
        solution = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times,
            rtol=1e-6,
            atol=1e-6
        )
        
        # 重构包含边界条件的完整解
        times = solution.t
        solutions = []
        
        for i in range(len(times)):
            u_internal = solution.y[:, i]
            u_full = np.zeros(self.nx)
            u_full[1:-1] = u_internal
            solutions.append(u_full)
        
        # 记录结束时间
        computation_time = time.time() - start_time
        
        # 准备结果字典
        results = {
            'times': times,
            'solutions': solutions,
            'computation_time': computation_time,
            'stability_parameter': None  # 自适应方法没有固定的稳定性参数
        }
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        参数:
            dt_explicit (float): 显式方法时间步长
            dt_implicit (float): 隐式方法时间步长
            dt_cn (float): Crank-Nicolson方法时间步长
            ivp_method (str): solve_ivp积分方法
            plot_times (list): 比较时间点
            
        返回:
            dict: 所有方法的结果
            
        实现步骤:
        1. 调用所有四种求解方法
        2. 记录计算时间和稳定性参数
        3. 返回比较结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 打印求解信息
        print("=" * 60)
        print(f"热传导方程求解比较 (L={self.L}, alpha={self.alpha}, nx={self.nx})")
        print("=" * 60)
        
        # 调用四种求解方法
        print("正在求解显式方法...")
        explicit_results = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        
        print("正在求解隐式方法...")
        implicit_results = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        
        print("正在求解Crank-Nicolson方法...")
        cn_results = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        
        print(f"正在求解solve_ivp方法 ({ivp_method})...")
        ivp_results = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        
        # 打印每种方法的计算时间和稳定性参数
        print("\n" + "=" * 60)
        print("计算时间和稳定性参数比较:")
        print(f"显式方法: 时间步长={dt_explicit}, 计算时间={explicit_results['computation_time']:.4f}秒, r={explicit_results['stability_parameter']:.4f}")
        print(f"隐式方法: 时间步长={dt_implicit}, 计算时间={implicit_results['computation_time']:.4f}秒, r={implicit_results['stability_parameter']:.4f}")
        print(f"Crank-Nicolson: 时间步长={dt_cn}, 计算时间={cn_results['computation_time']:.4f}秒, r={cn_results['stability_parameter']:.4f}")
        print(f"solve_ivp ({ivp_method}): 计算时间={ivp_results['computation_time']:.4f}秒, 自适应步长")
        print("=" * 60)
        
        # 返回所有结果的字典
        results = {
            'explicit': explicit_results,
            'implicit': implicit_results,
            'crank_nicolson': cn_results,
            'solve_ivp': ivp_results
        }
        
        return results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
            
        实现步骤:
        1. 创建2x2子图
        2. 为每种方法绘制不同时间的解
        3. 设置图例、标签和标题
        4. 可选保存图像
        """
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 方法名称和标签
        method_names = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        method_labels = ['显式方法', '隐式方法', 'Crank-Nicolson', 'solve_ivp']
        
        # 为每种方法绘制解曲线
        for i, method in enumerate(method_names):
            ax = axes[i]
            results = methods_results[method]
            
            # 绘制不同时间点的解
            for j, t in enumerate(results['times']):
                u = results['solutions'][j]
                ax.plot(self.x, u, label=f't={t:.1f}')
            
            # 设置标题、标签、图例
            ax.set_title(f'{method_labels[i]}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x,t)')
            ax.grid(True)
            ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 可选保存图像
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图像已保存为: {filename}")
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 参考方法
            
        返回:
            dict: 精度分析结果
            
        实现步骤:
        1. 选择参考解
        2. 计算其他方法与参考解的误差
        3. 统计最大误差和平均误差
        4. 返回分析结果
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            raise ValueError(f"参考方法 '{reference_method}' 不存在")
        
        # 获取参考解
        ref_results = methods_results[reference_method]
        ref_times = ref_results['times']
        ref_solutions = ref_results['solutions']
        
        # 计算各方法与参考解的误差
        accuracy_results = {}
        
        for method, results in methods_results.items():
            if method == reference_method:
                continue
                
            # 找到匹配的时间点
            method_times = results['times']
            method_solutions = results['solutions']
            
            errors = []
            for i, t in enumerate(method_times):
                # 找到参考解中最接近的时间点
                ref_idx = np.argmin(np.abs(np.array(ref_times) - t))
                ref_u = ref_solutions[ref_idx]
                method_u = method_solutions[i]
                
                # 计算误差
                error = np.abs(method_u - ref_u)
                errors.append(error)
            
            # 统计误差指标
            max_errors = [np.max(error) for error in errors]
            mean_errors = [np.mean(error) for error in errors]
            
            accuracy_results[method] = {
                'times': method_times,
                'max_errors': max_errors,
                'mean_errors': mean_errors,
                'average_max_error': np.mean(max_errors),
                'average_mean_error': np.mean(mean_errors)
            }
        
        # 打印精度分析结果
        print("\n" + "=" * 60)
        print("精度分析结果 (相对于参考方法 '{}'):".format(reference_method))
        print("=" * 60)
        
        for method, result in accuracy_results.items():
            print(f"{method}:")
            print(f"  平均最大误差: {result['average_max_error']:.6e}")
            print(f"  平均平均误差: {result['average_mean_error']:.6e}")
        
        return accuracy_results


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=21, T_final=25.0)
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.01,
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF'
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    # 返回结果
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
