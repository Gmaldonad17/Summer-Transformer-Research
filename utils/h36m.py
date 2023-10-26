import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import data_utils
from tqdm import tqdm
import pathlib
import math

def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray

class H36M(Dataset):

    def __init__(self, data_dir, window, tokens, output_t, skip_rate=1, actions=None, split=0, joints=32):
        """
        :param path_to_data:
        :param actions:
        :param output_t:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'h36m/dataset')
        self.split = split
        self.tokens = tokens
        self.window = window
        self.out_n = window * output_t
        self.in_n = window * tokens - self.out_n
        self.sample_rate = 2
        self.p3d = {}
        self.params = {}
        self.masks = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = [[1, 6, 7, 8, 9], [11], [5]]

        # self.action = -1

        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions


        if joints == 17:
            self.dim_used = np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42,
                 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83])
        elif joints == 22:
            self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                      26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                      46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                      75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        else:
            self.dim_used = np.arange(96)

        subs = subs[split]
        key = 0
        for subj in tqdm(subs):
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        # self.action = action_idx
                        the_sequence = readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        the_sequence[:, 0:6] = 0
                        p3d = data_utils.expmap2xyz_torch(the_sequence)

                        # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                        self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                        # self.p3d[key] = self.p3d[key][:, dim_used]

                        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)

                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_1 = [key] * len(valid_frames) #video_id
                        tmp_data_idx_2 = list(valid_frames) #valid_frame_id
                        tmp_data_idx_3 = list([action_idx]*len(valid_frames))
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3))
                        key += 1
                        # print(num_frames, len(valid_frames))
                else:
                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    # self.p3d[key] = self.p3d[key][:, dim_used]

                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_torch(the_seq2)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    # self.p3d[key + 1] = self.p3d[key + 1][:, dim_used]

                    fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                                                                   input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    tmp_data_idx_3 = list([action_idx]*len(valid_frames))
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    tmp_data_idx_3 = list([action_idx]*len(valid_frames))
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3))
                    key += 2

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame, action = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + (self.window * self.tokens) )

        pose = self.p3d[key][fs]
        observed = pose.copy() / 1000.0
        # observed = pose.copy()
        
        data = torch.tensor(observed[:, self.dim_used].reshape(observed.shape[0], -1, 3))
        Input = data[: self.in_n]

        if self.out_n == 0:
            output = Input
        else:
            Input = Input.view(-1, self.window, *Input.shape[1:])
            output =  data[self.in_n : ]
            output = output.view(-1, self.window, *output.shape[1:])

        return Input, output, action