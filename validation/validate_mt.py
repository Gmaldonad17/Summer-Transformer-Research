import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from models.models import *
from utils.metrics import *
from utils.h36m import H36M
from utils.logger import logger

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
    log = logger()
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
    parameters['MTc_PATH'] = []

    model = BlankModel(tgt_tokens, parameters['num_heads'], parameters['num_layers'], 
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
            n += batch[0].shape[0]
            
            Input, tgt, action = batch
            batches, tokens, window, keypoints, channels = Input.shape
            batches, tokentgt, window, keypoints, channels = tgt.shape
            []

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

            pck = calculate_pck(pred, tgt)
            mpjpe = calculate_mpjpe(pred, tgt, timings)
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
        
        log.write((running_loss, running_pck, running_mpjpe, running_fde))
        log.save()

if __name__ == '__main__':
    main()