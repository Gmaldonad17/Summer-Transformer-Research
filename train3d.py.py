import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from utils.h36m import H36M
from torch.utils.data import DataLoader
from models.models import BlankModel
from utils.trainer import ModelTrainer
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    datasets_parms = {
        'data_dir': "Datasets/",
        'window': 5,
        'tokens': 15,
        'output_t': 5,
        'joints': 22,
    }

    train_dataset = H36M(**datasets_parms, split=0)
    val_dataset = H36M(**datasets_parms, split=1)

    next_train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    next_val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

    for batch in next_val_dataloader:
        print(f"Input Shape:", batch[0].shape)
        print(f"Target Shape:", batch[1].shape)
        features = batch[1].shape
        break

    ###### Model Parameters ######
    batch_size, tgt_tokens, window, keypoints, channels = features
    parameters = load_parameters_from_txt("parameters.txt")
    print(parameters)

    current_time = time.strftime('%Y%m%d%H%M%S')
    folder_path = f"results/H{parameters['num_heads']}_L{parameters['num_layers']}_D{parameters['dropout']}_F{parameters['ffl']}_{current_time}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    model = BlankModel(tgt_tokens, parameters['num_heads'], parameters['num_layers'], 
                              parameters['ffl'], device, parameters['total_epochs'], dropout=parameters['dropout'][1])
    
    trainer = ModelTrainer(model, lr=0.001, epochs=parameters['total_epochs'], 
                           device=device, train_loader_len=len(next_train_dataloader), save_dir=folder_path)
    
    trainer.fit(next_train_dataloader, next_val_dataloader, loss_fn=torch.nn.MSELoss(reduction='none'))

if __name__ == '__main__':
    main()