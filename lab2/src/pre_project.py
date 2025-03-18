import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import os

def ensure_results_dir():
    """确保结果目录存在"""
    results_dir = '../lab2/results'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def load_data(data_path, file_prefix):
    """
    加载.npy或.mat格式的图像数据
    
    参数:
        data_path: 数据文件夹路径
        file_prefix: 文件名前缀（不含扩展名）
    
    返回:
        加载的图像数据，如果没有找到文件则返回None
    """
    npy_path = data_path / f"{file_prefix}.npy"
    mat_path = data_path / f"{file_prefix}.mat"
    
    if npy_path.exists():
        return np.load(npy_path)
    elif mat_path.exists():
        data = loadmat(mat_path)
        # 从mat文件中获取第一个非__开头的键值
        for key in data.keys():
            if not key.startswith('__'):
                return data[key]
        return None
    else:
        print(f"未找到{file_prefix}数据文件")
        return None

def display_image(img, title, window=None):
    """
    显示图像并调整窗口
    
    参数:
        img: 图像数据
        title: 图像标题
        window: 窗口值，形式为(窗宽, 窗位)的元组，如果为None则使用默认窗口
    """
    plt.figure(figsize=(8, 6))
    
    if window is not None:
        window_width, window_center = window
        vmin = window_center - window_width // 2
        vmax = window_center + window_width // 2
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"{title} - Window: Width={window_width}, Center={window_center}")
    else:
        plt.imshow(img, cmap='gray')
        plt.title(title)
    
    plt.colorbar(label='Intensity')
    return plt.gcf()

def save_image_with_various_windows(img, base_name, windows=None):
    """
    使用不同的窗口设置保存图像
    
    参数:
        img: 图像数据
        base_name: 基本文件名
        windows: 窗口设置列表，每项为(窗宽, 窗位, 描述)的元组
    """
    results_dir = ensure_results_dir()
    
    # 如果没有指定窗口设置，则使用默认设置
    if windows is None:
        # 默认窗口 - 显示完整范围
        fig = display_image(img, f"{base_name} - Default Window")
        fig.savefig(f"{results_dir}/{base_name}_default.png")
        plt.close(fig)
        return
    
    # 使用每个窗口设置保存图像
    for width, center, desc in windows:
        fig = display_image(img, f"{base_name} - {desc}", (width, center))
        fig.savefig(f"{results_dir}/{base_name}_{desc.replace(' ', '_').lower()}.png")
        plt.close(fig)

def process_medical_images():
    """处理CT和MRI医学图像"""
    data_path = Path('../data')
    
    # 加载CT数据
    ct_data = load_data(data_path, "lab2_CT")
    if ct_data is not None:
        print(f"CT数据形状: {ct_data.shape}, 数据类型: {ct_data.dtype}")
        print(f"CT数据范围: [{np.min(ct_data)}, {np.max(ct_data)}]")
        
        # CT窗口设置 - 根据不同应用调整
        ct_windows = [
            (400, 40, "Soft Tissue"),   # 软组织窗口
            (1500, -600, "Lung"),       # 肺窗口
            (2000, 400, "Bone"),        # 骨窗口
            (100, 40, "Brain")          # 脑窗口
        ]
        
        save_image_with_various_windows(ct_data, "CT", ct_windows)
    
    # 加载MRI数据
    mri_data = load_data(data_path, "lab2_MRI")
    if mri_data is not None:
        print(f"MRI数据形状: {mri_data.shape}, 数据类型: {mri_data.dtype}")
        print(f"MRI数据范围: [{np.min(mri_data)}, {np.max(mri_data)}]")
        
        # MRI窗口设置
        mri_windows = [
            (255, 127, "Standard"),     # 标准窗口
            (150, 100, "Brain Detail"),  # 脑细节窗口
            (300, 150, "Wide Range")     # 宽范围窗口
        ]
        
        save_image_with_various_windows(mri_data, "MRI", mri_windows)

if __name__ == "__main__":
    try:
        process_medical_images()
        print("医学图像处理完成")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
