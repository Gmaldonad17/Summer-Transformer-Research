import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_skeleton(skeletons, idx, fig_name):
    connections = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 12)
    ]

    fig = plt.figure()
    plt.get_current_fig_manager().window.wm_title(fig_name)
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
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_zlim(0, 1)

    plt.show(block=False)

def plot_connections(ax, skeleton, connections):
    # Plot connections between joints
    for connection in connections:
        joint1 = skeleton[connection[0]-1]
        joint2 = skeleton[connection[1]-1]
        xs = [joint1[0], joint2[0]]
        ys = [joint1[1], joint2[1]]
        zs = [joint1[2], joint2[2]]
        ax.plot(xs, ys, zs, c='b')

def calculate_pck(input_data, target_data, threshold_ratio=0.2):

    # Calculate Euclidean distance between input and target keypoints
    distances = torch.norm(input_data - target_data, dim=-1)

    # Calculate the bounding box corners
    top_left = torch.min(input_data, dim=2).values  # shape: (batch_size, window_size, xy)
    bottom_right = torch.max(input_data, dim=2).values  # shape: (batch_size, window_size, xy)

    # Calculate the skeleton diameter
    skeleton_diameter = torch.norm(top_left - bottom_right, dim=-1)  # shape: (batch_size, window_size)

    # Calculate the new threshold distance
    threshold_distance = threshold_ratio * skeleton_diameter
    threshold_distance = torch.clamp(threshold_distance, min=1e-7)  # prevent zero division
    threshold_distance = threshold_distance.unsqueeze(2).expand_as(distances)

    # Count the number of keypoints within the threshold distance
    correct_keypoints = torch.sum(distances <= threshold_distance, dim=1).sum().item()

    # Calculate the PCK
    total_keypoints = torch.numel(distances)
    pck = (correct_keypoints / total_keypoints) * 100.0

    return pck

def calculate_mpjpe(input_data, target_data, indices=[None]):
    results = []
    for index in indices:
        curr_input_data = input_data[:, :index, :, :].contiguous().view(-1, 3)
        curr_target_data = target_data[:, :index, :, :].contiguous().view(-1, 3)
        distances = torch.norm(curr_input_data - curr_target_data, p=2, dim=1)
        results.append(torch.mean(distances).item())
    return np.array(results)


def calculate_ade(input_data, target_data):
    diff = input_data - target_data
    dist = torch.norm(diff, dim=-1)
    return dist.mean()

def calculate_fde(input_data, target_data, indices=[None]):
    results = []
    for index in indices:
        curr_input_data = input_data[:, :index, :, :]
        curr_target_data = target_data[:, :index, :, :]
        distances = torch.norm(curr_input_data[:, -1:, :, :].contiguous().view(-1, 3) - \
                               curr_target_data[:, -1:, :, :].contiguous().view(-1, 3), p=2, dim=1)
        results.append(torch.mean(distances).item())
    return np.array(results)


def calculate_mAP(prediction, target, threshold):
    pred = torch.squeeze(prediction)
    tgt = torch.squeeze(target)

    # compute the norm for the last axis: (x,y,z) coordinates
    TP = torch.norm(pred-tgt, dim=-1) <= threshold
    TP = TP.int()
    FN = (~TP.bool()).int()

    # num_joints
    try:
        TP = torch.sum(TP, dim=1)
        FN = torch.sum(FN, dim=1)
    except:
        print("Shit")
        return 0

    # compute recall for each joint
    recall = TP.float() / (TP.float() + FN.float())
    # average over joints
    mAP = torch.mean(recall)

    return mAP, TP, FN