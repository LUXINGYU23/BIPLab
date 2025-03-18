import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import os
import sys

# 添加父目录到路径，以便导入 pre_project 中的函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pre_project import load_data, ensure_results_dir

def log_transform(image, v=1.0, c=1.0):
    """对数变换: s = c * log(1 + v*r)/log(1 + v)
    
    Args:
        image: 输入图像
        v: 变换参数，控制非线性程度
        c: 缩放常数，通常保持为1.0
        
    Returns:
        变换后的图像
    """
    # 确保图像值是非负的
    min_val = np.min(image)
    if min_val < 0:
        image = image - min_val
        
    # 归一化到[0, 1]区间
    normalized = image / np.max(image)
    
    # 应用对数变换 s = c * log(1 + v*r)/log(1 + v)
    # 注意：除以log(1 + v)是为了确保当r=1时，s=c
    result = c * np.log1p(v * normalized) / np.log1p(v)
    
    # 再次归一化结果到[0, 1]区间
    result = result / np.max(result)
    
    return result

def power_transform(image, gamma=0.5, c=1.0):
    """幂变换: s = c * r^gamma
    
    Args:
        image: 输入图像
        gamma: 幂参数，gamma < 1增强暗区，gamma > 1增强亮区
        c: 缩放常数
        
    Returns:
        变换后的图像
    """
    # 确保图像值是非负的
    min_val = np.min(image)
    if min_val < 0:
        image = image - min_val
    
    # 归一化到[0, 1]区间
    normalized = image / np.max(image)
    
    # 应用幂变换
    result = c * np.power(normalized, gamma)
    
    # 归一化结果到[0, 1]区间
    result = result / np.max(result)
    
    return result

def plot_histogram(image, title, ax=None):
    """绘制图像直方图
    
    Args:
        image: 输入图像
        title: 直方图标题
        ax: matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
    
    # 绘制直方图
    ax.bar((bins[:-1] + bins[1:])/2, hist, width=(bins[1]-bins[0]), alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')

def calculate_contrast(image):
    """计算图像的对比度
    
    Args:
        image: 输入图像
        
    Returns:
        对比度值 (标准差)
    """
    return np.std(image)

def calculate_entropy(image):
    """计算图像的信息熵
    
    Args:
        image: 输入图像
        
    Returns:
        信息熵值
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]  # 只考虑非零概率
    return entropy(hist)

def calculate_ssim(original, transformed):
    """计算结构相似度
    
    Args:
        original: 原始图像
        transformed: 变换后的图像
        
    Returns:
        SSIM值
    """
    return ssim(original, transformed, data_range=1.0)

def calculate_edge_content(image, sigma=1.0):
    """计算图像的边缘含量
    
    Args:
        image: 输入图像
        sigma: 高斯滤波的标准差
        
    Returns:
        边缘含量值
    """
    # 使用Sobel算子计算梯度
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # 返回平均梯度大小
    return np.mean(gradient)

def evaluate_transform(original, transformed):
    """评估变换效果
    
    Args:
        original: 原始图像
        transformed: 变换后的图像
        
    Returns:
        包含评估指标的字典
    """
    metrics = {
        'contrast': calculate_contrast(transformed),
        'contrast_ratio': calculate_contrast(transformed) / max(calculate_contrast(original), 1e-8),
        'entropy': calculate_entropy(transformed),
        'entropy_gain': calculate_entropy(transformed) - calculate_entropy(original),
        'ssim': calculate_ssim(original, transformed),
        'edge_content': calculate_edge_content(transformed),
        'edge_enhancement': calculate_edge_content(transformed) / max(calculate_edge_content(original), 1e-8)
    }
    
    # 计算总评分 (加权和)
    metrics['total_score'] = (
        0.25 * metrics['contrast_ratio'] +  # 对比度增益
        0.25 * min(2.0, max(0, metrics['entropy_gain'] * 2)) +  # 信息熵增益 (限制在0-2之间)
        0.25 * metrics['ssim'] +  # 结构相似性 (保持原始图像结构)
        0.25 * metrics['edge_enhancement']  # 边缘增强
    )
    
    return metrics

def find_best_transform(original, transforms):
    """找到最佳变换参数
    
    Args:
        original: 原始图像
        transforms: 变换结果字典 {参数名: 变换图像}
        
    Returns:
        最佳参数名称, 评估结果
    """
    best_param = None
    best_score = -float('inf')
    all_metrics = {}
    
    for param, img in transforms.items():
        metrics = evaluate_transform(original, img)
        all_metrics[param] = metrics
        
        if metrics['total_score'] > best_score:
            best_score = metrics['total_score']
            best_param = param
    
    return best_param, all_metrics

def main():
    """主函数：加载CT数据，执行强度变换，并保存结果"""
    data_path = Path('../lab2/data')
    
    # 加载CT数据
    ct_data = load_data(data_path, "lab2_CT")
    if ct_data is None:
        print("无法加载CT数据")
        return
    
    print(f"CT数据形状: {ct_data.shape}, 数据类型: {ct_data.dtype}")
    
    # 归一化原始图像到[0, 1]范围，以便比较
    original = ct_data / np.max(ct_data)
    
    # 应用更细粒度参数的对数变换（固定c=1.0，改变v）
    log_transforms = {
        'v=0.5': log_transform(ct_data, v=0.5),
        'v=1.0': log_transform(ct_data, v=1.0),
        'v=3.0': log_transform(ct_data, v=3.0),
        'v=5.0': log_transform(ct_data, v=5.0),
        'v=10.0': log_transform(ct_data, v=10.0),
        'v=20.0': log_transform(ct_data, v=20.0)
    }
    
    # 应用更细粒度参数的幂变换
    power_transforms = {
        'gamma=0.2': power_transform(ct_data, gamma=0.2),
        'gamma=0.3': power_transform(ct_data, gamma=0.3),
        'gamma=0.4': power_transform(ct_data, gamma=0.4),
        'gamma=0.6': power_transform(ct_data, gamma=0.6),
        'gamma=0.8': power_transform(ct_data, gamma=0.8),
        'gamma=1.2': power_transform(ct_data, gamma=1.2),
        'gamma=1.5': power_transform(ct_data, gamma=1.5)
    }
    
    # 自动找出最佳变换参数
    best_log_param, log_metrics = find_best_transform(original, log_transforms)
    best_power_param, power_metrics = find_best_transform(original, power_transforms)
    
    print(f"Best Log Transform Parameter: {best_log_param}")
    print(f"Best Power Transform Parameter: {best_power_param}")
    
    best_log = log_transforms[best_log_param]
    best_power = power_transforms[best_power_param]
    
    # 创建图像来显示所有对数变换结果 - 修复索引溢出问题
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('Comparison of Log Transformations on CT Image', fontsize=16)
    
    # 原始图像及其直方图
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].set_title('Original Histogram')
    plot_histogram(original, 'Original Histogram', axes[0, 1])
    
    # 因为我们有6个对数变换参数，按2行3列排列变换后的图像，对应的直方图在下方
    log_params = list(log_transforms.keys())
    for i, (title, img) in enumerate(log_transforms.items()):
        row = (i // 3) * 2  # 图像在0或2行
        col = i % 3 + 1     # 图像在1,2,3列
        
        # 变换后的图像
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Log Transform {title}')
        axes[row, col].axis('off')
        
        # 对应的直方图
        axes[row + 1, col].set_title(f'Histogram {title}')
        plot_histogram(img, f'Histogram {title}', axes[row + 1, col])
    
    # 如果有未使用的子图，隐藏它们
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:  # 原始图像
                continue
            if i == 0 and j == 1:  # 原始直方图
                continue
            if (i % 2 == 0) and j > 0 and (i // 2 * 3 + j - 1) < len(log_params):  # 变换图像
                continue
            if (i % 2 == 1) and j > 0 and (i // 2 * 3 + j - 1) < len(log_params):  # 变换直方图
                continue
            axes[i, j].axis('off')
            axes[i, j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    results_dir = ensure_results_dir()
    plt.savefig(f"{results_dir}/CT_log_transforms_comparison.png", dpi=300)
    plt.close()
    
    # 创建图像来显示所有幂变换结果 - 类似地修复索引问题
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('Comparison of Power Transformations on CT Image', fontsize=16)
    
    # 原始图像及其直方图
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].set_title('Original Histogram')
    plot_histogram(original, 'Original Histogram', axes[0, 1])
    
    # 幂变换有7个参数，分3行摆放
    power_params = list(power_transforms.keys())
    for i, (title, img) in enumerate(power_transforms.items()):
        row = (i // 3) * 2  # 图像在0或2行
        col = i % 3 + 1     # 图像在1,2,3列
        
        # 确保索引不超出范围
        if row >= 4:
            continue
            
        # 变换后的图像
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Power Transform {title}')
        axes[row, col].axis('off')
        
        # 对应的直方图
        axes[row + 1, col].set_title(f'Histogram {title}')
        plot_histogram(img, f'Histogram {title}', axes[row + 1, col])
    
    # 如果有未使用的子图，隐藏它们
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:  # 原始图像
                continue
            if i == 0 and j == 1:  # 原始直方图
                continue
            if (i % 2 == 0) and j > 0 and (i // 2 * 3 + j - 1) < len(power_params):  # 变换图像
                continue
            if (i % 2 == 1) and j > 0 and (i // 2 * 3 + j - 1) < len(power_params):  # 变换直方图
                continue
            axes[i, j].axis('off')
            axes[i, j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{results_dir}/CT_power_transforms_comparison.png", dpi=300)
    plt.close()
    
    # 绘制变换曲线图
    plt.figure(figsize=(16, 8))
    x = np.linspace(0, 1, 1000)
    
    # 对数变换曲线
    plt.subplot(1, 2, 1)
    log_v_values = [0.5, 1.0, 3.0, 5.0, 10.0, 20.0]
    for v in log_v_values:
        plt.plot(x, np.log1p(v * x) / np.log1p(v), label=f'v={v}')
    
    plt.title('Log Transform Curves Comparison')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 幂变换曲线
    plt.subplot(1, 2, 2)
    power_gamma_values = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    for gamma in power_gamma_values:
        plt.plot(x, x**gamma, label=f'gamma={gamma}')
    
    plt.title('Power Transform Curves Comparison')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/CT_transform_curves.png", dpi=300)
    plt.close()
    
    # 创建最佳对数和幂变换的对比图
    plt.figure(figsize=(18, 6))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original CT Image')
    plt.colorbar()
    plt.axis('off')
    
    # 最佳对数变换
    plt.subplot(1, 3, 2)
    plt.imshow(best_log, cmap='gray')
    plt.title(f'Best Log Transform ({best_log_param})')
    plt.colorbar()
    plt.axis('off')
    
    # 最佳幂变换
    plt.subplot(1, 3, 3)
    plt.imshow(best_power, cmap='gray')
    plt.title(f'Best Power Transform ({best_power_param})')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    results_dir = ensure_results_dir()
    plt.savefig(f"{results_dir}/CT_best_transformations.png", dpi=300)
    plt.close()
    
    # 创建评估指标比较图
    plt.figure(figsize=(20, 10))
    plt.suptitle('Evaluation Metrics for Different Parameters', fontsize=16)
    
    # 为评估指标创建子图
    metrics_to_plot = ['contrast_ratio', 'entropy_gain', 'ssim', 'edge_enhancement', 'total_score']
    metric_titles = ['Contrast Ratio', 'Entropy Gain', 'Structural Similarity', 'Edge Enhancement', 'Total Score']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        
        # 对数变换的指标
        log_params = list(log_metrics.keys())
        log_values = [log_metrics[p][metric] for p in log_params]
        
        # 幂变换的指标
        power_params = list(power_metrics.keys())
        power_values = [power_metrics[p][metric] for p in power_params]
        
        plt.bar(np.arange(len(log_params)), log_values, width=0.4, label='Log Transform', alpha=0.7)
        plt.bar(np.arange(len(power_params)) + 0.4, power_values, width=0.4, label='Power Transform', alpha=0.7)
        
        plt.xlabel('Parameters')
        plt.ylabel(metric_titles[i])
        plt.title(f'{metric_titles[i]} Comparison')
        
        if i == 0:  # 只在第一个子图中显示图例
            plt.legend()
        
        # 设置x轴刻度标签
        if len(log_params) == len(power_params):
            plt.xticks(np.arange(len(log_params)) + 0.2, [f"{lp} | {pp}" for lp, pp in zip(log_params, power_params)], rotation=45)
        else:
            plt.xticks(np.arange(max(len(log_params), len(power_params))) + 0.2, 
                      [f"{log_params[i] if i < len(log_params) else ''} | {power_params[i] if i < len(power_params) else ''}" 
                      for i in range(max(len(log_params), len(power_params)))], rotation=45)
        
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{results_dir}/CT_transform_metrics.png", dpi=300)
    plt.close()
    
    # 打印详细的评估结果
    print("\nDetailed Log Transform Metrics:")
    for param, metrics in log_metrics.items():
        print(f"  {param}: Score={metrics['total_score']:.4f}, Contrast={metrics['contrast_ratio']:.4f}, " +
              f"Entropy Gain={metrics['entropy_gain']:.4f}, SSIM={metrics['ssim']:.4f}, " +
              f"Edge Enhancement={metrics['edge_enhancement']:.4f}")
    
    print("\nDetailed Power Transform Metrics:")
    for param, metrics in power_metrics.items():
        print(f"  {param}: Score={metrics['total_score']:.4f}, Contrast={metrics['contrast_ratio']:.4f}, " +
              f"Entropy Gain={metrics['entropy_gain']:.4f}, SSIM={metrics['ssim']:.4f}, " +
              f"Edge Enhancement={metrics['edge_enhancement']:.4f}")
    
    print("""
    Intensity Transform Analysis Summary (Automated Parameter Selection):
    
    1. Log Transform (modified formula: s = c * log(1 + v*r)/log(1 + v)):
       - Best parameter: """ + best_log_param + """ automatically selected based on quantitative metrics
       - Effect of parameter v:
         * v=0.5: Slight non-linear enhancement
         * v=1.0: Standard logarithmic behavior
         * v=3.0: Moderate enhancement of dark areas
         * v=5.0: Stronger enhancement of dark areas
         * v=10.0: Very strong dark area enhancement
         * v=20.0: Extreme dark area enhancement, high non-linearity
       - Key metrics that influenced selection:
         * Contrast ratio: """ + f"{log_metrics[best_log_param]['contrast_ratio']:.4f}" + """
         * Entropy gain: """ + f"{log_metrics[best_log_param]['entropy_gain']:.4f}" + """
         * Structural similarity: """ + f"{log_metrics[best_log_param]['ssim']:.4f}" + """
         * Edge enhancement: """ + f"{log_metrics[best_log_param]['edge_enhancement']:.4f}" + """
         * Total score: """ + f"{log_metrics[best_log_param]['total_score']:.4f}" + """
       
    2. Power Transform:
       - Best parameter: """ + best_power_param + """ automatically selected based on quantitative metrics
       - Key metrics that influenced selection:
         * Contrast ratio: """ + f"{power_metrics[best_power_param]['contrast_ratio']:.4f}" + """
         * Entropy gain: """ + f"{power_metrics[best_power_param]['entropy_gain']:.4f}" + """
         * Structural similarity: """ + f"{power_metrics[best_power_param]['ssim']:.4f}" + """
         * Edge enhancement: """ + f"{power_metrics[best_power_param]['edge_enhancement']:.4f}" + """
         * Total score: """ + f"{power_metrics[best_power_param]['total_score']:.4f}" + """
       
    3. Comparing Log and Power Transforms:
       - The modified log transform with optimal v parameter allows more flexible control over non-linearity
       - Power transform with gamma < 1 tends to enhance dark regions more uniformly
       - The optimal parameters were selected to balance contrast enhancement, information content,
         structural preservation and edge visibility
    """)

if __name__ == "__main__":
    main()
