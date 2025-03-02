import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from pathlib import Path

def process_image_scaling(img):
    try:
        results_dir = '../lab1/result'
        os.makedirs(results_dir, exist_ok=True)
        
        N_values = [2, 4, 8]
        x_position = 100
        original_shape = img.shape
        
        plt.rcParams['figure.dpi'] = 300
        fig = plt.figure(figsize=(18, 5*len(N_values)))
        gs = plt.GridSpec(len(N_values), 2, figure=fig, hspace=0.15, wspace=0.12)
        
        for idx, N in enumerate(N_values):
            # Downsample and enlarge, ensure size matching
            reduced = zoom(img, 1/N)
            reduced_display = zoom(reduced, (original_shape[0]/reduced.shape[0], 
                                          original_shape[1]/reduced.shape[1]))
            enlarged = zoom(reduced, (original_shape[0]/reduced.shape[0], 
                                   original_shape[1]/reduced.shape[1]))
            
            # Ensure all image sizes match
            assert img.shape == reduced_display.shape == enlarged.shape, f"Image sizes don't match: {img.shape}, {reduced_display.shape}, {enlarged.shape}"
            
            # Get profile lines
            original_line = img[:, x_position]
            reduced_line_index = int(round(x_position / N))
            reduced_line = reduced[:, reduced_line_index]
            enlarged_line = enlarged[:, x_position]
            
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.imshow(np.vstack([img, reduced_display, enlarged]), cmap='gray', 
                      interpolation='nearest')
            ax1.axvline(x=x_position, color='r', linestyle='--', linewidth=1.5)
            ax1.set_title(f'N={N}: Original (top) vs. Reduced (middle) vs. Enlarged (bottom)', 
                         pad=10, fontsize=11)
            ax1.set_xticks([])  
            ax1.set_yticks([])  
            
            ax2 = fig.add_subplot(gs[idx, 1])
            ax2.plot(original_line, label='Original', linewidth=1.8)
            ax2.plot(np.linspace(0, len(original_line)-1, len(reduced_line)), 
                    reduced_line, label='Reduced', linestyle='--', linewidth=1.8)
            ax2.plot(enlarged_line, label='Enlarged', linestyle='-.', linewidth=1.8)
            ax2.legend(fontsize=9, loc='best')
            ax2.set_title(f'N={N}: X=100 Line Profile Comparison', 
                         pad=10, fontsize=11)
            ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        output_path = os.path.join(results_dir, 'scaling_comparison_multiple.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        process_image_scaling(img)
    except Exception as e:
        print(f"Program execution error: {str(e)}")
