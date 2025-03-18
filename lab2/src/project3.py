import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import os
import sys

# 添加父目录到路径，以便导入 pre_project 中的函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pre_project import load_data, ensure_results_dir

def compute_histogram(image, n_bins=256):
    """
    计算图像的直方图
    
    参数:
        image: 输入图像
        n_bins: 直方图的柱数
        
    返回:
        hist: 直方图
        bins: 柱的边界
    """
    # 如果图像不是uint8类型，先转换为0-255范围
    if image.dtype != np.uint8:
        # 计算图像的最小值和最大值
        min_val = np.min(image)
        max_val = np.max(image)
        
        # 归一化到[0, 255]
        img_normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        img_normalized = image
    
    # 计算直方图
    hist, bins = np.histogram(img_normalized.flatten(), bins=n_bins, range=(0, 255))
    
    return hist, bins

def histogram_equalization(image):
    """
    执行直方图均衡化
    
    参数:
        image: 输入图像
        
    返回:
        equalized_image: 均衡化后的图像
    """
    # 确保图像是uint8类型或将其转换为uint8类型
    if image.dtype != np.uint8:
        min_val = np.min(image)
        max_val = np.max(image)
        image_uint8 = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()
    
    # 获取图像尺寸和像素总数
    height, width = image_uint8.shape
    num_pixels = height * width
    
    # 计算直方图
    hist, _ = compute_histogram(image_uint8)
    
    # 计算累积分布函数 (CDF)
    cdf = np.cumsum(hist)
    
    # 归一化CDF到[0, 255]
    cdf_normalized = (cdf * 255 / num_pixels).astype(np.uint8)
    
    # 使用映射关系创建均衡化后的图像
    equalized_image = cdf_normalized[image_uint8]
    
    return equalized_image

def plot_image_and_histogram(image, title, ax_img, ax_hist):
    """
    在指定的轴上绘制图像及其直方图
    
    参数:
        image: 输入图像
        title: 图像标题
        ax_img: 用于绘制图像的轴
        ax_hist: 用于绘制直方图的轴
    """
    # 绘制图像
    img_display = ax_img.imshow(image, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')
    plt.colorbar(img_display, ax=ax_img, orientation='vertical', fraction=0.046, pad=0.04)
    
    # 计算并绘制直方图
    if image.dtype != np.uint8:
        min_val = np.min(image)
        max_val = np.max(image)
        img_normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        img_normalized = image
    
    hist, bins = np.histogram(img_normalized.flatten(), bins=256, range=(0, 255))
    
    # 绘制直方图
    ax_hist.bar(np.arange(256), hist, width=1, alpha=0.7)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_title(f"Histogram of {title}")
    ax_hist.set_xlabel("Pixel Value")
    ax_hist.set_ylabel("Frequency")
    # 添加网格线使直方图更易读
    ax_hist.grid(True, alpha=0.3)

def calculate_metrics(original, equalized):
    """
    计算均衡化前后的图像指标
    
    参数:
        original: 原始图像
        equalized: 均衡化后的图像
        
    返回:
        metrics: 包含各种指标的字典
    """
    # 确保图像是uint8类型
    if original.dtype != np.uint8:
        min_val = np.min(original)
        max_val = np.max(original)
        original_uint8 = ((original - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        original_uint8 = original.copy()
    
    if equalized.dtype != np.uint8:
        min_val = np.min(equalized)
        max_val = np.max(equalized)
        equalized_uint8 = ((equalized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        equalized_uint8 = equalized.copy()
    
    # 计算原始图像的直方图和均衡化图像的直方图
    hist_orig, _ = np.histogram(original_uint8.flatten(), bins=256, range=(0, 255))
    hist_equalized, _ = np.histogram(equalized_uint8.flatten(), bins=256, range=(0, 255))
    
    # 计算非零灰度级数量
    non_zero_bins_orig = np.sum(hist_orig > 0)
    non_zero_bins_equalized = np.sum(hist_equalized > 0)
    
    # 计算对比度（用灰度级的标准差作为对比度的度量）
    contrast_orig = np.std(original_uint8)
    contrast_equalized = np.std(equalized_uint8)
    
    # 计算熵（信息量）
    # 计算概率分布
    p_orig = hist_orig / np.sum(hist_orig)
    p_equalized = hist_equalized / np.sum(hist_equalized)
    
    # 去掉零概率以避免log(0)
    p_orig = p_orig[p_orig > 0]
    p_equalized = p_equalized[p_equalized > 0]
    
    # 计算熵
    entropy_orig = -np.sum(p_orig * np.log2(p_orig))
    entropy_equalized = -np.sum(p_equalized * np.log2(p_equalized))
    
    # 计算均匀度（使用直方图变异系数的倒数作为均匀度的度量）
    cv_orig = np.std(hist_orig) / (np.mean(hist_orig) if np.mean(hist_orig) > 0 else 1)
    cv_equalized = np.std(hist_equalized) / (np.mean(hist_equalized) if np.mean(hist_equalized) > 0 else 1)
    
    uniformity_orig = 1 / (1 + cv_orig)  # 归一化到(0,1]
    uniformity_equalized = 1 / (1 + cv_equalized)
    
    return {
        "非零灰度级数量": {"原图": non_zero_bins_orig, "均衡化后": non_zero_bins_equalized},
        "对比度": {"原图": contrast_orig, "均衡化后": contrast_equalized},
        "熵": {"原图": entropy_orig, "均衡化后": entropy_equalized},
        "均匀度": {"原图": uniformity_orig, "均衡化后": uniformity_equalized}
    }

def analyze_histogram_equalization(original, equalized):
    """
    分析直方图均衡化效果并生成摘要
    
    参数:
        original: 原始图像
        equalized: 均衡化后的图像
        
    返回:
        analysis: 分析结果文本
    """
    metrics = calculate_metrics(original, equalized)
    
    # 生成分析结果
    analysis = "直方图均衡化分析:\n\n"
    
    # 非零灰度级数量
    orig_bins = metrics["非零灰度级数量"]["原图"]
    eq_bins = metrics["非零灰度级数量"]["均衡化后"]
    bin_change = eq_bins - orig_bins
    bin_change_percent = (bin_change / orig_bins * 100) if orig_bins > 0 else float('inf')
    
    analysis += f"1. 灰度级利用率:\n"
    analysis += f"   - 原图使用了256级中的{orig_bins}级 ({orig_bins/2.56:.1f}%)\n"
    analysis += f"   - 均衡化后使用了256级中的{eq_bins}级 ({eq_bins/2.56:.1f}%)\n"
    analysis += f"   - {'增加' if bin_change > 0 else '减少'}了{abs(bin_change)}级 "
    analysis += f"({abs(bin_change_percent):.1f}%)\n\n"
    
    # 对比度
    orig_contrast = metrics["对比度"]["原图"]
    eq_contrast = metrics["对比度"]["均衡化后"]
    contrast_change_percent = ((eq_contrast - orig_contrast) / orig_contrast * 100) if orig_contrast > 0 else float('inf')
    
    analysis += f"2. 图像对比度:\n"
    analysis += f"   - 原图对比度: {orig_contrast:.2f}\n"
    analysis += f"   - 均衡化后对比度: {eq_contrast:.2f}\n"
    analysis += f"   - 对比度{'增加' if eq_contrast > orig_contrast else '减少'}了{abs(contrast_change_percent):.1f}%\n\n"
    
    # 熵
    orig_entropy = metrics["熵"]["原图"]
    eq_entropy = metrics["熵"]["均衡化后"]
    entropy_change_percent = ((eq_entropy - orig_entropy) / orig_entropy * 100) if orig_entropy > 0 else float('inf')
    
    analysis += f"3. 信息熵 (图像含有信息量):\n"
    analysis += f"   - 原图熵: {orig_entropy:.4f} bits\n"
    analysis += f"   - 均衡化后熵: {eq_entropy:.4f} bits\n"
    analysis += f"   - 熵{'增加' if eq_entropy > orig_entropy else '减少'}了{abs(entropy_change_percent):.1f}%\n\n"
    
    # 均匀度
    orig_uniformity = metrics["均匀度"]["原图"]
    eq_uniformity = metrics["均匀度"]["均衡化后"]
    uniformity_change_percent = ((eq_uniformity - orig_uniformity) / orig_uniformity * 100) if orig_uniformity > 0 else float('inf')
    
    analysis += f"4. 直方图均匀度:\n"
    analysis += f"   - 原图均匀度: {orig_uniformity:.4f}\n"
    analysis += f"   - 均衡化后均匀度: {eq_uniformity:.4f}\n"
    analysis += f"   - 均匀度{'增加' if eq_uniformity > orig_uniformity else '减少'}了{abs(uniformity_change_percent):.1f}%\n\n"
    
    # 总结
    analysis += "5. 总结:\n"
    if eq_contrast > orig_contrast and eq_entropy > orig_entropy:
        analysis += "   - 直方图均衡化成功提高了图像对比度和信息量\n"
        analysis += "   - 灰度级分布更加均匀，使用了更多的可用灰度级\n"
        analysis += "   - 增强了图像中的细节和结构，特别是在原本对比度较低的区域\n"
    elif eq_contrast > orig_contrast:
        analysis += "   - 均衡化主要提高了图像对比度\n"
        analysis += "   - 某些区域的细节可能因为对比度增强而更加明显\n"
    elif eq_entropy > orig_entropy:
        analysis += "   - 均衡化主要增加了图像的信息量\n"
        analysis += "   - 灰度级分布更加均匀，但对比度提升有限\n"
    else:
        analysis += "   - 均衡化效果有限，可能原图已经有较好的对比度和灰度分布\n"
        analysis += "   - 考虑使用自适应直方图均衡化或其他增强方法\n"
    
    return analysis

def main():
    """主函数：加载CT数据，执行直方图均衡化，并分析结果"""
    data_path = Path('../lab2/data')
    
    # 加载CT数据
    ct_data = load_data(data_path, "lab2_CT")
    if ct_data is None:
        print("无法加载CT数据")
        return
    
    print(f"CT数据形状: {ct_data.shape}, 数据类型: {ct_data.dtype}")
    
    # 执行直方图均衡化
    equalized_image = histogram_equalization(ct_data)
    
    # 创建图像来展示原始图像、均衡化后的图像及其直方图
    fig = plt.figure(figsize=(16, 12))
    
    # 创建子图网格：2行2列
    gs = fig.add_gridspec(2, 2)
    
    # 原始图像
    ax_orig_img = fig.add_subplot(gs[0, 0])
    ax_orig_hist = fig.add_subplot(gs[1, 0])
    plot_image_and_histogram(ct_data, "Original CT Image", ax_orig_img, ax_orig_hist)
    
    # 均衡化后的图像
    ax_eq_img = fig.add_subplot(gs[0, 1])
    ax_eq_hist = fig.add_subplot(gs[1, 1])
    plot_image_and_histogram(equalized_image, "Histogram Equalized Image", ax_eq_img, ax_eq_hist)
    
    # 调整布局并保存
    plt.tight_layout()
    results_dir = ensure_results_dir()
    plt.savefig(f"{results_dir}/ct_histogram_equalization.png", dpi=300)
    plt.close()
    
    # 创建详细的CDF比较图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 计算原始图像的直方图和CDF
    hist_orig, _ = compute_histogram(ct_data)
    cdf_orig = np.cumsum(hist_orig)
    cdf_orig_normalized = cdf_orig / cdf_orig.max() * 100  # 归一化为百分比
    
    # 计算均衡化图像的直方图和CDF
    hist_eq, _ = compute_histogram(equalized_image)
    cdf_eq = np.cumsum(hist_eq)
    cdf_eq_normalized = cdf_eq / cdf_eq.max() * 100  # 归一化为百分比
    
    # 绘制直方图对比
    axes[0].bar(np.arange(256), hist_orig, alpha=0.5, label='Original', color='blue', width=1)
    axes[0].bar(np.arange(256), hist_eq, alpha=0.5, label='Equalized', color='red', width=1)
    axes[0].set_title('Histogram Comparison')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制CDF对比
    axes[1].plot(np.arange(256), cdf_orig_normalized, label='Original CDF', color='blue', linewidth=2)
    axes[1].plot(np.arange(256), cdf_eq_normalized, label='Equalized CDF', color='red', linewidth=2)
    axes[1].set_title('Cumulative Distribution Function (CDF)')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Cumulative Percentage (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 理想线性CDF参考线
    axes[1].plot([0, 255], [0, 100], 'g--', label='Ideal Linear CDF', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/ct_histogram_cdf_comparison.png", dpi=300)
    plt.close()
    
    # 分析均衡化结果
    analysis = analyze_histogram_equalization(ct_data, equalized_image)
    print("\n" + analysis)
    
    # 保存分析结果到文本文件
    with open(f"{results_dir}/histogram_equalization_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis)
    
    print(f"直方图均衡化分析结果已保存到 {results_dir}/histogram_equalization_analysis.txt")
    
if __name__ == "__main__":
    main()
