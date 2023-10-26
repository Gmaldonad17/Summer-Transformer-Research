import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import torch

import torch.multiprocessing as mp

from utils.datasets import TokenActionDataset, set_seed
from models.models import *

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

def draw_skeleton(skeleton, img_size):
    """Draws a skeleton on a white image."""
    img = np.ones((*img_size, 3)) * 255
    for point in skeleton:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)
    return img.astype(np.uint8)

def crop_img(img, x_cut, y_cut, mid_cut):
    img = img[:, x_cut:-x_cut]
    img = img[y_cut:-y_cut]
    img = np.concatenate((img[:mid_cut], img[-mid_cut:]), axis=0)

    return img

def create_skeleton_video(skeleton_data0, skeleton_data1, output_file='skeleton.mp4', fps=30):
    """Generates a video from a sequence of skeletons."""
    # Get video parameters
    num_frames = skeleton_data0.shape[0]
    img_size = (1080, 1920)
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (512, 512))

    for i in tqdm(range(num_frames)):
        # Draw the skeleton
        skeleton0 = skeleton_data0[i].reshape(-1, 2)
        skeleton1 = skeleton_data1[i].reshape(-1, 2)

        img0 = draw_skeleton(skeleton0, img_size)
        img1 = draw_skeleton(skeleton1, img_size)
        
        img = np.vstack((img0, img1))
        img = crop_img(img, 500, 300, 450)
        
        img = cv2.resize(img, (512, 512))
        # Write the frame to the video
        video.write(img)

    # Close the video writer
    video.release()

def main():
    seed = 42
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    window_size = 20
    stride = 10
    body_type = 'rgb_body'
    summary_csv_path = "summary_pixel.csv"
    skeleton_dataset = TokenActionDataset(summary_csv_path, window_size=window_size, stride=stride, body_type=body_type)
    
    edge_index.to(device)

    features = None
    batch = skeleton_dataset.__getitem__(0)[1]
    features = batch.shape

    ###### Model Parameters ######
    # num_nodes = 16
    window, decoding_size, channels = features
    laten_space = 80
    
    PATH = f"models/trained/3metric/GAE_2010s3{laten_space}.pth"

    # Load back the same model for Testing 
    encoder = GCNEncoder(input_dim=channels, window_dim=window, hidden_dim=16, output_dim=32, fc_output_dim=laten_space)
    decoder = GCNDecoder(output_dim=channels, hidden_dim=16, window_dim=window, decoding_size=decoding_size, input_dim=32, laten_space=laten_space)
    auto_encoder = GCNAutoEncoder(encoder, decoder, edge_index, device=device).to(device)
    auto_encoder.load_state_dict(torch.load(PATH))
    auto_encoder.eval()

    NOR_MAX = torch.Tensor([[[2516.143, 1463.549]]]).to(device)
    NOR_MIN = torch.Tensor([[[-100.1795, -344.0357]]]).to(device)

    data_range = range(160, 180)
    unencoded_list = []
    decoded_list = []

    for i in data_range:
        Input, Target, _ = skeleton_dataset.__getitem__(i)
        unencoded_list.append(Input)
        decoded_list.append(Target)

    Input = torch.stack(unencoded_list, dim=0)
    Target = torch.stack(decoded_list, dim=0)
    
    Input = Input.to(device)
    Target = Target.to(device)
    
    # Forward pass
    with torch.no_grad():
        decoded_skel = auto_encoder(Input)

    decoded_skel = decoded_skel * (NOR_MAX - NOR_MIN) + NOR_MIN
    Target = Target * (NOR_MAX - NOR_MIN) + NOR_MIN

    flattened_decoded = decoded_skel.reshape(-1, 25, 2).cpu().numpy()
    flattened_target = Target.reshape(-1, 25, 2).cpu().numpy()

    # Create the video
    create_skeleton_video(flattened_target, flattened_decoded, output_file=f"video_samples/GAE_2010r2{laten_space}.mp4")


if __name__ == '__main__':
    main()