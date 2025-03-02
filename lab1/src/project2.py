import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.patches as patches

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

def add_magnified_region(ax, img, rect_coords, zoom_loc, zoom_size):
    """
    Add a magnified region of the image to show detail
    
    Parameters:
    - ax: matplotlib axis to draw on
    - img: image array
    - rect_coords: (x, y, width, height) of the region to magnify
    - zoom_loc: (x, y) location to place the zoomed region
    - zoom_size: (width, height) of the zoomed region
    """
    # Draw rectangle to indicate the magnified region
    rect = patches.Rectangle(
        (rect_coords[0], rect_coords[1]), 
        rect_coords[2], rect_coords[3], 
        linewidth=1, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Extract the region to magnify
    x, y, w, h = rect_coords
    region = img[y:y+h, x:x+w]
    
    # Add an inset zoom axes
    zoom_ax = ax.inset_axes([zoom_loc[0], zoom_loc[1], zoom_size[0], zoom_size[1]])
    zoom_ax.imshow(region, cmap='gray')
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    zoom_ax.spines['bottom'].set_color('red')
    zoom_ax.spines['top'].set_color('red') 
    zoom_ax.spines['right'].set_color('red')
    zoom_ax.spines['left'].set_color('red')

def display_multiple_levels(img):
    try:
        results_dir = '../lab1/result'
        os.makedirs(results_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        levels = [8, 6, 4]  # Corresponding to 256, 64, 16 levels
        
        # Select a region to magnify (adjust these coordinates based on your image)
        h, w = img.shape
        rect_coords = (w//3, h//3, w//10, h//10)  # (x, y, width, height)
        
        for ax, n in zip(axes, levels):
            reduced = reduce_intensity_levels(img, n)
            ax.imshow(reduced, cmap='gray')
            ax.set_title(f'{2**n} levels')
            ax.axis('off')
            
            # Add magnified region to show detail differences
            add_magnified_region(
                ax, reduced, 
                rect_coords=rect_coords,
                zoom_loc=(-0.15, 0.65),  # Relative position in the axes
                zoom_size=(0.3, 0.3)    # Relative size in the axes
            )
        
        plt.tight_layout()
        output_path = os.path.join(results_dir, 'intensity_levels_with_magnification.png')
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
