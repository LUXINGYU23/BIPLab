import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import os
import sys

# 添加父目录到路径，以便导入 pre_project 中的函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pre_project import load_data, ensure_results_dir

def apply_affine_transform(image, transform_matrix):
    """使用仿射变换矩阵对图像进行变换
    
    Args:
        image: 输入图像
        transform_matrix: 3x3仿射变换矩阵
        
    Returns:
        变换后的图像
    """
    height, width = image.shape
    result = np.zeros_like(image)
    
    # 对每个目标像素进行反向映射
    for y in range(height):
        for x in range(width):
            # 构建目标位置的齐次坐标
            position = np.array([x, y, 1])
            
            # 应用逆变换矩阵获得源位置
            inv_matrix = np.linalg.inv(transform_matrix)
            src_x, src_y, _ = inv_matrix @ position
            
            # 检查是否在原图范围内
            if 0 <= src_y < height and 0 <= src_x < width:
                # 双线性插值
                y1, y2 = int(np.floor(src_y)), int(np.ceil(src_y))
                x1, x2 = int(np.floor(src_x)), int(np.ceil(src_x))
                
                if y2 >= height: y2 = height - 1
                if x2 >= width: x2 = width - 1
                
                # 计算插值权重
                if y1 == y2: wy2 = 0
                else: wy2 = (src_y - y1) / (y2 - y1)
                wy1 = 1 - wy2
                
                if x1 == x2: wx2 = 0
                else: wx2 = (src_x - x1) / (x2 - x1)
                wx1 = 1 - wx2
                
                # 插值计算新像素值
                result[y, x] = (image[y1, x1] * wy1 * wx1 +
                                image[y1, x2] * wy1 * wx2 +
                                image[y2, x1] * wy2 * wx1 +
                                image[y2, x2] * wy2 * wx2)
    
    return result

def translate_image(image, shift_y, shift_x=0):
    """使用变换矩阵进行平移
    
    Args:
        image: 输入图像
        shift_y: 垂直方向的位移量(正值向下，负值向上)
        shift_x: 水平方向的位移量(正值向右，负值向左)，默认为0
        
    Returns:
        平移后的图像
    """
    # 创建平移矩阵
    transform_matrix = np.array([
        [1, 0, shift_x],
        [0, 1, shift_y],
        [0, 0, 1]
    ])
    
    return apply_affine_transform(image, transform_matrix)

def rescale_image(image, scale_factor=0.5):
    """使用变换矩阵进行缩放，保持中心点不变
    
    Args:
        image: 输入图像
        scale_factor: 缩放比例
        
    Returns:
        缩放后的图像
    """
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    
    # 创建缩放矩阵，需要先平移到原点，缩放，再平移回中心点
    transform_matrix = np.array([
        [scale_factor, 0, center_x * (1 - scale_factor)],
        [0, scale_factor, center_y * (1 - scale_factor)],
        [0, 0, 1]
    ])
    
    return apply_affine_transform(image, transform_matrix)

def flip_image(image, axis='x'):
    """使用变换矩阵沿指定轴翻转图像
    
    Args:
        image: 输入图像
        axis: 'x'表示水平翻转，'y'表示垂直翻转
        
    Returns:
        翻转后的图像
    """
    height, width = image.shape
    
    if axis.lower() == 'x':
        # 沿x轴翻转（上下翻转）
        transform_matrix = np.array([
            [1, 0, 0],
            [0, -1, height-1],
            [0, 0, 1]
        ])
    elif axis.lower() == 'y':
        # 沿y轴翻转（左右翻转）
        transform_matrix = np.array([
            [-1, 0, width-1],
            [0, 1, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("轴参数必须是 'x' 或 'y'")
    
    return apply_affine_transform(image, transform_matrix)

def rotate_image(image, angle_rad):
    """使用变换矩阵旋转图像，围绕图像中心点
    
    Args:
        image: 输入图像
        angle_rad: 旋转角度（弧度）
        
    Returns:
        旋转后的图像
    """
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    
    # 计算旋转矩阵参数
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # 创建旋转矩阵，需要先平移中心点到原点，旋转，再平移回来
    transform_matrix = np.array([
        [cos_theta, -sin_theta, -center_x * cos_theta + center_y * sin_theta + center_x],
        [sin_theta, cos_theta, -center_x * sin_theta - center_y * cos_theta + center_y],
        [0, 0, 1]
    ])
    
    return apply_affine_transform(image, transform_matrix)

def composite_transform(image):
    """组合变换：使用矩阵乘法实现多重变换的组合
    
    Args:
        image: 输入图像
        
    Returns:
        组合变换后的图像
    """
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    
    # 1. 向下平移20像素的矩阵
    translate_matrix = np.array([
        [1, 0, 0],
        [0, 1, 20],
        [0, 0, 1]
    ])
    
    # 2. 缩放到0.5倍的矩阵
    scale_matrix = np.array([
        [0.5, 0, center_x * 0.5],
        [0, 0.5, center_y * 0.5],
        [0, 0, 1]
    ])
    
    # 3. 沿x轴翻转的矩阵
    flip_matrix = np.array([
        [1, 0, 0],
        [0, -1, height-1],
        [0, 0, 1]
    ])
    
    # 4. 旋转π/4的矩阵
    angle = np.pi/4
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotate_matrix = np.array([
        [cos_theta, -sin_theta, -center_x * cos_theta + center_y * sin_theta + center_x],
        [sin_theta, cos_theta, -center_x * sin_theta - center_y * cos_theta + center_y],
        [0, 0, 1]
    ])
    
    # 按顺序组合变换矩阵（最后应用的变换矩阵放在最左边）
    # 最终矩阵 = 旋转矩阵 * 翻转矩阵 * 缩放矩阵 * 平移矩阵
    composite_matrix = rotate_matrix @ flip_matrix @ scale_matrix @ translate_matrix
    
    # 应用组合变换矩阵
    return apply_affine_transform(image, composite_matrix)

def main():
    """主函数：加载MRI数据，执行所有变换，并保存结果"""
    data_path = Path('../lab2/data')
    
    # 加载MRI数据
    mri_data = load_data(data_path, "lab2_MRI")
    if mri_data is None:
        print("无法加载MRI数据")
        return
    
    print(f"MRI数据形状: {mri_data.shape}, 数据类型: {mri_data.dtype}")
    
    # 执行各种仿射变换
    translated_down = translate_image(mri_data, 20)  # 向下平移50像素
    # translated_up = translate_image(mri_data, -50)  # 向上平移50像素
    rescaled = rescale_image(mri_data)  # 缩放到原尺寸的一半
    flipped_x = flip_image(mri_data, 'x')  # 沿x轴翻转
    # flipped_y = flip_image(mri_data, 'y')  # 沿y轴翻转
    rotated = rotate_image(mri_data, np.pi/4)  # 旋转π/4
    composite = composite_transform(mri_data)  # 组合变换
    
    # 创建一个大图来显示所有结果
    fig = plt.figure(figsize=(18, 12))
    
    # 原图
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(mri_data, cmap='gray')
    ax1.set_title('Original MRI Image')
    # 显示坐标轴而不是关闭
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 向下平移
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(translated_down, cmap='gray')
    ax2.set_title('Downward Translation (50px)')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # # 向上平移
    # ax3 = fig.add_subplot(2, 4, 3)
    # im3 = ax3.imshow(translated_up, cmap='gray')
    # ax3.set_title('Upward Translation (50px)')
    # ax3.set_xlabel('X-axis')
    # ax3.set_ylabel('Y-axis')
    # plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 缩放
    ax4 = fig.add_subplot(2, 3, 3)
    im4 = ax4.imshow(rescaled, cmap='gray')
    ax4.set_title('Rescaled to Half Size')
    ax4.set_xlabel('X-axis')
    ax4.set_ylabel('Y-axis')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 沿x轴翻转
    ax5 = fig.add_subplot(2, 3, 4)
    im5 = ax5.imshow(flipped_x, cmap='gray')
    ax5.set_title('Flipped along X-axis')
    ax5.set_xlabel('X-axis')
    ax5.set_ylabel('Y-axis')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # # 沿y轴翻转
    # ax6 = fig.add_subplot(2, 4, 6)
    # im6 = ax6.imshow(flipped_y, cmap='gray')
    # ax6.set_title('Flipped along Y-axis')
    # ax6.set_xlabel('X-axis')
    # ax6.set_ylabel('Y-axis')
    # plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # 旋转
    ax7 = fig.add_subplot(2, 3, 5)
    im7 = ax7.imshow(rotated, cmap='gray')
    ax7.set_title('Rotated π/4')
    ax7.set_xlabel('X-axis')
    ax7.set_ylabel('Y-axis')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # 组合变换
    ax8 = fig.add_subplot(2, 3, 6)
    im8 = ax8.imshow(composite, cmap='gray')
    ax8.set_title('Composite Transform')
    ax8.set_xlabel('X-axis')
    ax8.set_ylabel('Y-axis')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # 保存大图
    results_dir = ensure_results_dir()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/MRI_affine_transformations.png", dpi=300)
    plt.close()
    
    # 单独保存每个变换结果
    save_single_images = False
    if save_single_images:
        plt.figure()
        plt.imshow(translated_down, cmap='gray')
        plt.title('Downward Translation (50px)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_translated_down.png")
        plt.close()
        
        plt.figure()
        plt.imshow(translated_up, cmap='gray')
        plt.title('Upward Translation (50px)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_translated_up.png")
        plt.close()
        
        plt.figure()
        plt.imshow(rescaled, cmap='gray')
        plt.title('Rescaled to Half Size')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_rescaled.png")
        plt.close()
        
        plt.figure()
        plt.imshow(flipped_x, cmap='gray')
        plt.title('Flipped along X-axis')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_flipped_x.png")
        plt.close()
        
        plt.figure()
        plt.imshow(flipped_y, cmap='gray')
        plt.title('Flipped along Y-axis')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_flipped_y.png")
        plt.close()
        
        plt.figure()
        plt.imshow(rotated, cmap='gray')
        plt.title('Rotated π/4')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_rotated.png")
        plt.close()
        
        plt.figure()
        plt.imshow(composite, cmap='gray')
        plt.title('Composite Transform')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.colorbar(label='Intensity')
        plt.savefig(f"{results_dir}/MRI_composite.png")
        plt.close()
    
    print("仿射变换完成，结果已保存")

if __name__ == "__main__":
    main()
