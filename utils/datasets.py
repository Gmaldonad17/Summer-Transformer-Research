import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import threading

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AutoEncoderDataset(Dataset):
    def __init__(self, csv_file, window_size, stride, body_type):
        # Initialize class attributes
        self.body_type = body_type
        self.window_size = window_size
        self.stride = stride
        
        # Read the CSV file into a pandas DataFrame
        self.data_summary = pd.read_csv(csv_file)
        
        # Extract frame counts and clip values to ensure they are non-negative
        frame_counts = self.data_summary.iloc[:, 3:].values
        values_clipped = np.clip((frame_counts - (self.window_size + 1)) // stride * stride, a_min=0, a_max=None)

        # Compute cumulative frame counts
        self.cumulative_frame_counts = np.cumsum(np.sum(values_clipped, axis=1))
        
        self.NOR_MAX = np.array([[[-3.141314,  -3.1263762, -3.115997 ]]]) # xyz
        self.NOR_MIN = np.array([[[3.1402893, 3.1250648, 3.114822 ]]])


    def __len__(self):
        return int(self.cumulative_frame_counts[-1] / self.stride)
    
    def __getitem__(self, idx):

        idx *= self.stride
        npy_file_idx = np.searchsorted(self.cumulative_frame_counts, idx)
        
        file_path = self.data_summary.iloc[npy_file_idx, 0]
        data = np.load(file_path, allow_pickle=True)
        action_class = self.data_summary.iloc[npy_file_idx, 1]

        shift = self.cumulative_frame_counts[npy_file_idx - 1] if npy_file_idx > 0 else 0

        idx -= shift

        skeletons = data[idx : idx + self.window_size]
        
        # # X' = (X - X_min) / (X_max - X_min) Normalizing
        skeletons = np.clip((skeletons - self.NOR_MIN) / (self.NOR_MAX - self.NOR_MIN), 0, 1)

        Input = torch.Tensor(skeletons)
        Input = torch.nan_to_num(Input, nan=0)

        # start_idx = (self.window_size - self.stride) // 2
        # end_idx = start_idx + self.stride
        Target = Input # [start_idx:end_idx]

        return (Input, Target, action_class)


class NextTokenDataset(Dataset):
    def __init__(self, csv_file, window_size, stride, max_tokens, tgt_tokens, body_type):
        # Initialize class attributes
        self.body_type = body_type
        self.window_size = window_size
        self.stride = stride
        self.max_tokens = max_tokens
        self.tgt_tokens = tgt_tokens
        self.required_frames = self.max_tokens * self.stride - self.stride + self.window_size
        
        # Read the CSV file into a pandas DataFrame
        self.data_summary = pd.read_csv(csv_file)
        
        # Extract frame counts and clip values to ensure they are non-negative
        frame_counts = self.data_summary.iloc[:, 3:].values
        values_clipped = np.clip((frame_counts - self.required_frames) // stride * stride, a_min=0, a_max=None)

        # Compute cumulative frame counts
        self.cumulative_frame_counts = [
            np.cumsum(np.sum(values_clipped, axis=1)),
        ]
        
        # Compute intermediate values and append to cumulative_frame_counts
        tmp = np.insert(self.cumulative_frame_counts[0], 0, 0)[:-1].reshape(-1, 1)
        self.cumulative_frame_counts.append(np.cumsum(values_clipped, axis=1) + tmp)
        del tmp

        # Define normalization maximum and minimum values
        self.NOR_MAX = np.array([[[-3.823892, -2.716611, 0.0]]]) # xyz
        self.NOR_MIN = np.array([[[4.169164, 2.312103, 6.191546]]])
        # self.NOR_MAX = np.array([[[2516.143, 1463.549]]])
        # self.NOR_MIN = np.array([[[-100.1795, -344.0357]]])


    def __len__(self):
        return int(self.cumulative_frame_counts[0][-1] / self.stride)
    
    def __getitem__(self, idx):
        
        idx *= self.stride
        npy_file_idx = np.searchsorted(self.cumulative_frame_counts[0], idx)
        person_idx = np.searchsorted(self.cumulative_frame_counts[1][npy_file_idx], idx)
        
        file_path = self.data_summary.iloc[npy_file_idx, 0]
        data = np.load(file_path, allow_pickle=True)[0]
        action_class = self.data_summary.iloc[npy_file_idx, 1]

        if person_idx > 0:
            shift = self.cumulative_frame_counts[1][npy_file_idx][person_idx - 1]
        else:
            shift = self.cumulative_frame_counts[0][npy_file_idx - 1] if npy_file_idx > 0 else 0

        idx -= shift

        skeletons = data[self.body_type + str(person_idx)][idx : idx + self.required_frames] # [:,:,:2]
        
        # # X' = (X - X_min) / (X_max - X_min) Normalizing
        skeletons = np.clip((skeletons - self.NOR_MIN) / (self.NOR_MAX - self.NOR_MIN), 0, 1)

        # Use numpy stride_tricks to create the windows
        shape = (self.max_tokens, self.window_size, *skeletons.shape[1:])
        strides = (skeletons.strides[0] * self.stride, *skeletons.strides[:])
        Input = np.lib.stride_tricks.as_strided(skeletons, shape=shape, strides=strides)

        Input = torch.Tensor(Input)
        Input = torch.nan_to_num(Input, nan=0)

        start_idx = (self.window_size - self.stride) // 2
        end_idx = start_idx + self.stride
        Target = Input[-self.tgt_tokens:]
        Input = Input[:-self.tgt_tokens]

        return (Input, Target, action_class)

class SkeletonDataset(Dataset):
    def __init__(self, csv_file, window_size=1, stride=1, parse=1):
        self.window_size = window_size
        self.stride = stride
        self.parse = parse
        
        self.data_summary = pd.read_csv(csv_file)
        self.cumulative_frame_counts = np.cumsum(self.data_summary['frame_count'].values - (self.window_size + 1))
        

    def __len__(self):
        last_value = self.cumulative_frame_counts[-1]
        
        output = int( int(str(last_value)[:2] + '0' * (len(str(last_value)) - 2)) / (self.parse * self.stride) )
        return output

    def __getitem__(self, idx):
        # Random Index 
        
        idx *= self.stride
        
        # Find the index of the npy file that contains the skeleton with the given index
        npy_file_idx = np.searchsorted(self.cumulative_frame_counts, idx - 1)
        
        file_path = self.data_summary.iloc[npy_file_idx, 0]

        # Load the npy file
        data = np.load(file_path, allow_pickle=True)

        # Calculate the index of the skeleton within the npy file
        skel_idx = idx - self.cumulative_frame_counts[npy_file_idx - 1] if npy_file_idx > 0 else idx

        # Extract skel_body0 from the data
        angles_body0 = data[0]['angles_body0'][skel_idx]
        
        # Convert skel_body0 to a torch tensor
        angles_body0_tensor = torch.tensor(angles_body0, dtype=torch.float32).view(-1, 1)
        
        
        for i in range(self.window_size - 1):
            body = torch.tensor(data[0]['angles_body0'][skel_idx + i], dtype=torch.float32).view(-1, 1)
            angles_body0_tensor = np.concatenate((angles_body0_tensor, body), axis=1)
        
        angles_body0_tensor /= 180

        return (angles_body0_tensor, angles_body0_tensor[:, 10:-10])