import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from models.models import *
from utils.datasets import NextTokenDataset, set_seed
from utils.metrics import *
from utils.h36m import H36M

def plot_3d_skeleton(skeleton):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from the skeleton
    x = skeleton[:, 0]
    y = skeleton[:, 1]
    z = skeleton[:, 2]

    # Plot each point as a dot with its index labeled
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], label=str(i))

    # Add labels for each point in the plot
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], str(i), fontsize=10, color='black')

    # Set axis labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal limits for all axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

def main():
    datasets_parms = {
        'data_dir': "Datasets/",
        'window': 5,
        'tokens': 15,
        'output_t': 5,
        'joints': 22,
    }

    test_dataset = H36M(**datasets_parms, split=2)
    next_test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)

    for batch in next_test_dataloader:
        print(f"Input Shape:", batch[0].shape)
        print(f"Target Shape:", batch[1].shape)
        features = batch[1].shape
        break

    ###### Model Parameters ######
    batch_size, tgt_tokens, window, keypoints, channels = features

    parameters = load_parameters_from_txt("parameters.txt")
    # T_PATH = 'models/trained/transformer/'
    # T_NAME = f"Start_Token_EDT_{datasets_parms['tokens']}_{parameters['num_heads']}{parameters['num_layers']}_{parameters['ffl']}{parameters['total_epochs']}_3"
    # parameters['AE_PATH'] = f"models/trained/autoencoder/3metric/GAE_T.pth"
    # parameters['AE_PATH'] = None
    parameters['MTc_PATH'] = "/home/adaneshp/pose_prediction/pose_prediction/results/H33_L8_D[0, 0.2]_F512_20230819152903/14.pt"
    # print(T_NAME)

    # _, model = LoadModel(**parameters)
    model = normalTransformer(tgt_tokens, parameters['num_heads'], parameters['num_layers'], 
                              parameters['ffl'], device, parameters['total_epochs'], dropout=parameters['dropout'][1]).to(device)
    model.load_state_dict(torch.load(parameters['MTc_PATH']))

    timings = [2, 8, 14, 18, 22, None]
    dataloaders = [next_test_dataloader]

    for loader in dataloaders:

        running_loss = 0
        running_pck = 0
        running_mpjpe = np.zeros((len(timings)))
        running_ade = np.zeros(2)
        running_fde = np.zeros((len(timings)))
        running_mAP = np.zeros(2)

        n = 0
        for batch in tqdm(loader):
            # mean = torch.load("mean-std/human3.6_mean.pt")
            # std = torch.load("mean-std/human3.6_std.pt")
            # batch[0] = (batch[0] - mean.expand((batch[0].shape)))/std.expand((batch[0].shape))
            # batch[1] = (batch[1] - mean.expand((batch[1].shape)))/std.expand((batch[1].shape))
            n += batch[0].shape[0]
            
            Input, tgt, action = batch
            batches, tokens, window, keypoints, channels = Input.shape
            batches, tokentgt, window, keypoints, channels = tgt.shape
            Input = Input.to(model.device).view(batches, tokens * window, keypoints * channels)
            tgt = tgt.to(model.device).view(batches, tokentgt * window, keypoints, channels)

            # start_token = torch.zeros((batches, 1, keypoints * channels), device=model.device)
            start_token = action.unsqueeze(1).unsqueeze(2).expand(-1, 1, keypoints*channels).to(model.device)
            start_Input = torch.cat((start_token, Input), dim=1)
            
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                pred = model.infer(start_Input, tokentgt * window, action=action)
                pred = pred.view(batches, tokentgt * window, keypoints, channels)

            # Calculate the loss using the mask
            
            loss_fn = torch.nn.MSELoss(reduction='none')
            
            loss = loss_fn(tgt, pred).mean().item()

            pred *= 1000
            tgt *= 1000
            # tgt = model.embed(tgt)
            
            
            

            # pred = (pred * std.expand((pred.shape))) + mean.expand((pred.shape))
            # tgt = (tgt * std.expand((tgt.shape))) + mean.expand((tgt.shape))

            pck = calculate_pck(pred, tgt)
            mpjpe = calculate_mpjpe(pred, tgt, timings)
            # ade = calculate_ade(pred, tgt)
            fde = calculate_fde(pred, tgt, timings)

            running_loss += loss
            running_pck += pck
            running_mpjpe += mpjpe
            running_fde += fde

        
        loss_mean = running_loss / len(loader)
        pck_mean = running_pck / len(loader)

        print(f"Testing Loss: {loss_mean}")
        print("Testing mpjpe: " + ', '.join(f'{value:.8f}' for value in (running_mpjpe / len(loader))))
        print(f"Testing pck: {pck_mean}")
        print("Testing fde: " + ', '.join(f'{value:.8f}' for value in (running_fde / len(loader))))
        # print("Testing mAP: " + ', '.join(f'{value:.8f}' for value in (running_mAP / len(loader)).flatten()))

if __name__ == '__main__':
    main()