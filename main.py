import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from powspec2d import continuous_fourier_2d_torch
import cv2
from tqdm import tqdm
import glob
import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO)

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')


parser = argparse.ArgumentParser(description='Pointset visualization')
parser.add_argument('--pointset_name', type=str, default="blue_n1024", help='point set name')
args = parser.parse_args()


def plot_point_np(pts, cls, size, title):
    plt.figure(1)
    plt.scatter(pts[:, 0], pts[:, 1], s=size, c=cls)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def visualize_pointset():

    """ file """
    
    output_folder = 'results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    """ parameters setup """
    # pointset_name = f"blue_n1024"
    # pointset_name = f"blue_n2048"
    # pointset_name = f"white_n1024"
    # pointset_name = f"white_n2048"
    # pointset_name = f"green_n1024"
    # pointset_name = f"green_n2048"
    # pointset_name = f"pink_n1024"
    # pointset_name = f"pink_n2048"
    pointset_name = args.pointset_name
    
    pointset_path = f"data/{pointset_name}.txt"

    num_points = int(pointset_name.split('_')[1][1:])

    if pointset_name.split('_')[0] == "white":
        samples = np.random.rand(num_points, 2).astype(np.float32)
    else:
        samples = np.loadtxt(pointset_path).astype(np.float32)

    
    if num_points == 1024:
        dot_size = 10
    elif num_points == 2048:
        dot_size = 7    # hard-coded
    else:
        raise ValueError('pointset_name not found')
    
    plot_point_np(samples, 'black', dot_size, f"results/{pointset_name}")

    res = 256
    N = samples.shape[0]
    tar_spectrum_2d = continuous_fourier_2d_torch(device, torch.from_numpy(samples).float().to(device), res, freqStep=1)
    
    if True:
        plt.figure(1)
        plt.imshow(tar_spectrum_2d.detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        # plt.show()
        plt.savefig('results/tar_spectrum_2d', bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close('all')

    print('Done.')


def main():
    visualize_pointset()


if __name__ == '__main__':
    main()

