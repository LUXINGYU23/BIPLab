import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_color_overlay(img, threshold=200):
    try:
        # Normalize to uint8 type
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Generate mask
        mask = img > threshold
        
        # Create RGB image by copying grayscale to all channels
        color_img = np.stack([img, img, img], axis=-1)
        # Set masked regions to pure red
        color_img[mask] = [255, 0, 0]
        
        # Visualize original image, mask and overlay effect
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title('Original Image')
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f'Mask (Threshold={threshold})')
        ax[2].imshow(color_img)
        ax[2].set_title('Red Highlight Overlay')
        
        # Save results
        output_dir = Path('../lab1/result')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'correct_color_overlay.png', bbox_inches='tight')
        plt.close()
        print(f"Results saved to: {output_dir / 'correct_color_overlay.png'}")
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Load data
        data_path = Path('../lab1/data')
        if (data_path / 'lab1.npy').exists():
            img = np.load(data_path / 'lab1.npy')
        elif (data_path / 'lab1.mat').exists():
            from scipy.io import loadmat
            img = loadmat(data_path / 'lab1.mat')['image']
        else:
            raise FileNotFoundError("Could not find lab1.npy or lab1.mat")
        create_color_overlay(img, threshold=150)
    except Exception as e:
        print(f"Main error: {str(e)}")