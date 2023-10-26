import numpy as np
import os
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_skeleton(skeleton):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Extract x and y coordinates from the skeleton
    x = skeleton[:, 0]
    y = skeleton[:, 1]

    # Plot each point as a dot with its index labeled
    for i in range(len(x)):
        ax.scatter(x[i], y[i], label=str(i))

    # Add labels for each point in the plot
    for i in range(len(x)):
        ax.text(x[i], y[i], str(i), fontsize=10, color='black')

    # Set axis labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set equal limits for both axes
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)

    plt.show()


if __name__ == '__main__':

    data_path = "M:/Datasets/h36m/dataset"
    output_path = "M:/Datasets/h36m/numpy"
    required_keypoints = [6, 7, 8, 9, 10, 11, 12, 13, 14, 
                      15, 16, 17, 21, 22, 23, 24, 25, 
                      26, 27, 28, 29, 30, 31, 32, 36, 
                      37, 38, 39, 40, 41, 42, 43, 44, 
                      45, 46, 47, 51, 52, 53, 54, 55, 
                      56, 57, 58, 59, 63, 64, 65, 66, 
                      67, 68, 75, 76, 77, 78, 79, 80, 
                      81, 82, 83, 87, 88, 89, 90, 91, 92]

    data_dirs = [f"{data_path}/{x}"
            for x in os.listdir(data_path)]

    for d in data_dirs[1:]:
        dirs = [f"{d}/{x}" for x in os.listdir(d)]
        dir_path = d.split('/')[-3:][-1]

        out_path = f"{output_path}/{dir_path}"
        os.makedirs(out_path, exist_ok=True)
        for dd in dirs:
            print('[ANMG/D] Loading file:', dd)
            txt_mat = np.loadtxt(dd, delimiter=',').astype(np.float32)
            # Downsample video and only accept specific keypoints
            txt_mat = txt_mat[:, required_keypoints][0::2]
            txt_mat = txt_mat.reshape(txt_mat.shape[0], -1, 3)
            txt_mat = txt_mat[:, :, [2,1]]
            
            plot_2d_skeleton(txt_mat[0])

            in_file = dd.split('/')[-1].split('.')[0]
            out_file = f"{out_path}/{in_file}.npy"
            print('[ANMG/D] Saving to:', out_file)
            # np.save(out_file, txt_mat)