#!/usr/bin/env python
# coding=utf-8

'''
transform the skeleton data in NTU RGB+D dataset into the numpy arrays for a more efficient data loading
'''

import numpy as np
from tqdm import tqdm
import os
import sys 

user_name = 'user'
save_npy_path = 'raw_npy/'
load_txt_path = 'raw_txt/'
missing_file_path = 'ntu120_missing.txt'
step_ranges = list(range(0,100)) # just parse range, for the purpose of paralle running. 


toolbar_width = 50
def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write('\n')

def load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 

def read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    with open(file_path, 'r') as f:
        datas = f.readlines()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    bodymat = {'file_name': file_path[-29:-9], 'nbodys': [], 'njoints': njoints}

    for body in range(max_body):
        if save_skelxyz:
            bodymat[f'skel_body{body}'] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat[f'rgb_body{body}'] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat[f'depth_body{body}'] = np.zeros(shape=(nframe, njoints, 2))
    
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = f'skel_body{body}'
            rgb_body = f'rgb_body{body}'
            depth_body = f'depth_body{body}'
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame, joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame, joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame, joint] = jointinfo[5:7]
    # prune the abundant bodys 
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat[f'skel_body{each}']
            if save_rgbxy:
                del bodymat[f'rgb_body{each}']
            if save_depthxy:
                del bodymat[f'depth_body{each}']
    return bodymat


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 * magnitude_v2 == 0:
        return 0

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
    return angle

def compute_angles(skeletons, angle_dict):
    batch_size, joint_number, xyz = skeletons.shape
    num_angles = len(angle_dict)
    angles_output = np.zeros((batch_size, num_angles))

    for i, angle_name in enumerate(angle_dict):
        joint_idx1, joint_idx2, joint_idx3, joint_idx4 = angle_dict[angle_name]

        for j in range(batch_size):
            v1 = skeletons[j, joint_idx1-1] - skeletons[j, joint_idx2-1]
            v2 = skeletons[j, joint_idx3-1] - skeletons[j, joint_idx4-1]
            angle = angle_between_vectors(v1, v2)
            angles_output[j, i] = angle

    return angles_output


def rotate_skeleton(skeleton, hip_joints, point_index, angle_degrees=90):
    # Adjust hip_joints and point_index for zero-based indexing
    hip_joints[0] -= 1
    hip_joints[1] -= 1
    point_index -= 1

    # Extract the hip joint points from the skeleton
    hip_points = skeleton[hip_joints]

    # Extract the point to rotate around from the skeleton
    point_to_rotate_around = skeleton[point_index]

    # Translate the hip points so that the point_to_rotate_around is at the origin
    hip_points -= point_to_rotate_around

    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Construct the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1]])

    # Rotate the hip points
    rotated_hip_points = np.dot(rotation_matrix, hip_points.T).T

    # Translate the hip points back to their original position
    rotated_hip_points += point_to_rotate_around

    # Update the hip joint points in the skeleton
    skeleton[hip_joints] = rotated_hip_points

    # Return the rotated skeleton
    return skeleton


def compute_angles_dim(skeletons, angle_dim, hip):
    batch_size, joint_number, xyz = skeletons.shape
    num_angles = len(angle_dim)
    angles_output_x = np.zeros((batch_size, num_angles))
    angles_output_z = np.zeros((batch_size, num_angles))
    angles_output = [angles_output_x, angles_output_z]

    for r in range(2):
        for i, angle_name in enumerate(angle_dim):
            joint_idx1, joint_idx2 = angle_dim[angle_name]
            joint_idx3, joint_idx4 = hip[0::2]

            if angle_name[0] == 'l':
                joint_idx3, joint_idx4 = hip[0::2][::-1]

            for j in range(batch_size):
                if r:
                    skeletons[j] = rotate_skeleton(skeletons[j].copy(), hip[0::2], hip[1])

                v1 = skeletons[j, joint_idx1-1] - skeletons[j, joint_idx2-1]
                v2 = skeletons[j, joint_idx3-1] - skeletons[j, joint_idx4-1]
                angle = angle_between_vectors(v1, v2)
                angles_output[r][j, i] = angle

    stacked = np.stack((angles_output[0], angles_output[1]), axis=-1)
    interleaved = stacked.reshape(stacked.shape[0], -1)
    
    return interleaved



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_skeleton(skeletons, idx):
    connections = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 12)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    skeleton = skeletons[idx]

    # Plot the joints as points with labels
    for i, joint in enumerate(skeleton):
        ax.scatter(joint[0], joint[1], joint[2], c='r', marker='o')
        ax.text(joint[0], joint[1], joint[2], str(i+1))

    # Plot connections between joints
    plot_connections(ax, skeleton, connections)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def plot_connections(ax, skeleton, connections):
    # Plot connections between joints
    for connection in connections:
        joint1 = skeleton[connection[0]-1]
        joint2 = skeleton[connection[1]-1]
        xs = [joint1[0], joint2[0]]
        ys = [joint1[1], joint2[1]]
        zs = [joint1[2], joint2[2]]
        ax.plot(xs, ys, zs, c='b')


def main():
    missing_files = load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path)
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = {filename: True for filename in alread_exist}
    #J1 - J2 | J3 - J4 J2 and J4 center
    angles_dict = {'rShoulder':[10,9,5,9], 'lShoulder':[6,5,9,5], 
                    'rElbow':[11,10,9,10], 'lElbow':[7,6,5,6],
                    'rHip':[18,17,13,17], 'lHip':[14,13,17,13],
                    'rKnee':[19,18,17,18], 'lKnee':[15,14,13,14]}
    
    angles_dim = {'rShoulder':[10,9], 'lShoulder':[6,5], 
                    'rElbow':[11,10], 'lElbow':[7,6],
                    'rHip':[18,17], 'lHip':[14,13],
                    'rKnee':[19,18], 'lKnee':[15,14]}
    
    hip_dim = [13, 1, 17]


    for each in tqdm(datalist):
        S = int(each[1:4])
        if S not in step_ranges:
            continue
        if f"{each}.skeleton.npy" in alread_exist_dict:
            continue
        if each[:20] in missing_files:
            continue
        loadname = os.path.join(load_txt_path, each)
        mat = [read_skeleton(loadname)]
        mat = np.array(mat) 

        keys = list(mat[0].keys())
        for key in keys:
            match_key = 'skel_body'
            if key[:len(match_key)] == match_key:
                n = key[len(match_key):]
                skeleton = mat[0][key]
                output_angles = compute_angles(skeleton, angles_dict)
                output_angles_dim = compute_angles_dim(skeleton, angles_dim, hip_dim)
                
                mat[0]['angles_body' + n] = output_angles
                mat[0]['angle_dim_body' + n] = output_angles_dim

        save_path = os.path.join(save_npy_path, f"{each.split('.')[0]}.npy")
        np.save(save_path, mat)
        # raise ValueError()

    end_toolbar()

if __name__ == '__main__':
    main()