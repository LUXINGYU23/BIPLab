import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def reduce_intensity_levels(img, n):
    """
    Reduce image to 2^n gray levels
    """
    levels = 2 ** n
    # Calculate actual gray range of the image
    gray_range = img.max() - img.min() + 1
    # Normalize grayscale to 0~(levels-1), round and map back to 0~255
    quantized = np.floor((img / gray_range * (levels - 1)) + 0.5)
    return (quantized / (levels - 1) * 255).astype(np.uint8)

def display_multiple_levels(img):
    try:
        results_dir = '../lab1/result'
        os.makedirs(results_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        levels = [8, 6, 4]  # Corresponding to 256, 64, 16 levels
        
        for ax, n in zip(axes, levels):
            reduced = reduce_intensity_levels(img, n)
            ax.imshow(reduced, cmap='gray')
            ax.set_title(f'{2**n} levels')
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(results_dir, 'intensity_levels.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Image saved successfully: {output_path}")
    except Exception as e:
        print(f"Failed to process image: {str(e)}")

if __name__ == '__main__':
    try:
        data_path = Path('../lab1/data')
        img = None
        
        if (data_path / 'lab1.npy').exists():
            img = np.load(data_path / 'lab1.npy')
        elif (data_path / 'lab1.mat').exists():
            from scipy.io import loadmat
            img = loadmat(data_path / 'lab1.mat')['image']
        else:
            raise FileNotFoundError("lab1.npy or lab1.mat file not found")
        
        if img is not None:
            display_multiple_levels(img)
    except Exception as e:
        print(f"Program execution error: {str(e)}")
