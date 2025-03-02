import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_results_dir():
    results_dir = '../lab1/result'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_circle_image(size=256):
    img = np.zeros((size, size))
    center = size // 2
    radius = size // 4
    thickness = 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = np.abs(dist - radius) <= thickness
    img[mask] = 255
    return img

def create_diagonal_gradient(size=256):
    img = np.zeros((size, size))
    np.fill_diagonal(np.fliplr(img), 255)
    y, x = np.ogrid[:size, :size]
    dist_to_diagonal = np.abs(y + x - (size - 1))
    max_dist = size - 1
    normalized_dist = dist_to_diagonal / max_dist
    img = 255 * (1 - normalized_dist)
    
    return img

def save_results(img, filename):
    try:
        results_dir = ensure_results_dir()
        output_path = os.path.join(results_dir, f'{filename}.png')
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray', extent=[0, 255, 0, 255])
        plt.colorbar(label='Intensity')
        plt.xlabel('X Coordinate (0-255)')
        plt.ylabel('Y Coordinate (0-255)')
        plt.title(f'{filename.replace("_", " ").title()}')
        
        plt.savefig(output_path)
        plt.close()
        print(f"Image saved successfully: {output_path}")
    except Exception as e:
        print(f"Failed to save image: {str(e)}")

if __name__ == '__main__':
    try:
        circle_img = create_circle_image()
        gradient_img = create_diagonal_gradient()
        
        save_results(circle_img, 'circle')
        save_results(gradient_img, 'diagonal_gradient')
    except Exception as e:
        print(f"Program execution error: {str(e)}")
