import torch
import math
import numpy as np
from tqdm import tqdm
from torch import nn

import torch.nn.functional as F
[]

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

edge_index = torch.tensor([
    [11, 10], [10, 9], [9, 12], [12, 13], [15, 14], [15, 16],
    [9, 17], [17, 18], [18, 20], [20, 19], [20, 21],
    [9, 8], [8, 0], [0, 1], [1, 2], [2, 3],
    [8, 4], [4, 5], [5, 6], [6, 7]
], dtype=torch.long).t().contiguous()

# Determine the number of nodes (max node index + 1)
num_nodes = edge_index.max().item() + 1

# Create tensor with self-attachments
self_attachments = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)

# Concatenate self_attachments with edge_index
edge_index = torch.cat([edge_index, self_attachments], dim=1).to(device)
    
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, max_len=51):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class BlankModel(nn.Module):
    def __init__(self, embedding, decoding, edge_index, 
                 tgt_tokens, num_heads, num_layers, 
                 ffl, device, total_epochs, dropout=0.0):
        super().__init__()
        self.device = device
        self.tgt_tokens = tgt_tokens
        
        []

        self.latent_space = embedding.fc.out_features
        self.num_heads = num_heads
        self.ffl = ffl

        self.trained_epochs = 0

        self.mode_split = [0.75, 0.75]
        self.training_mode = 0
        self.total_epochs = total_epochs
            
        []
    
    def blank(self, x):
        []
    
    def transform(self, tgt, x, tgt_mask=None):
        []
    
    def blank(self, x):
        []
    
    def blank(self, x):
        []

    def forward(self, x, tgt, tgt_mask=None):
        []
        return x
    
    def infer(self, x, seq_len):
        start_token = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=self.device)
        []
        start_token = self.[](start_token)
        pred = torch.empty((x.shape[0], 0, *x.shape[2:]), device=self.device)

        for _ in range(seq_len):
            out = self.[](start_token, x) 
            predt1 = out[:, -1:]
            predt1 = self.[](predt1)
            start_token = torch.cat((start_token, predt1), dim=1)
            pred = torch.cat((pred, predt1), dim=1)

        pred = self.[](pred)

        return pred
    
    def create_tgt_mask(self, seq_len):
        []


    def training_step(self, batch, optimizer, loss_fn, epoch):
        Input, tgt, _ = batch
        Input = Input.to(self.device)
        tgt = tgt.to(self.device)
        []

        self.train()
        
        # Zero the gradients of the model parameters
        optimizer.zero_grad()
        self.trained_epochs = epoch + 1
        epochs_division = (self.trained_epochs / self.total_epochs)
        
        if  epochs_division > self.mode_split[0] and self.training_mode != 1:
            []

        # For teacher forcing, just use the ground truth as tgt
        []

        loss = loss_fn(pred, tgt).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item(), 0.0
    
    
    def validation_step(self, batch, loss_fn):
        []
    

def LoadModel(tgt_tokens, window, keypoints, 
              channels, dims, laten_space, 
              dropout, num_heads, num_layers, 
              ffl, total_epochs,
              AE_PATH=None, MTc_PATH=None):

    [] = [](input_dim=channels, window_dim=window, hidden_dim=dims[0], output_dim=dims[1], fc_output_dim=laten_space, dropout=0.0)
    [] = [](output_dim=channels, hidden_dim=dims[0], window_dim=window, decoding_size=keypoints, input_dim=dims[1], laten_space=laten_space, dropout=0.0)
    []

    MTc = None

    if AE_PATH != None:
        [].load_state_dict(torch.load(AE_PATH))

        [] = [](auto_encoder.encoder, auto_encoder.decoder, edge_index, 
                tgt_tokens, num_heads=num_heads, num_layers=num_layers, 
                ffl=ffl, device=device, total_epochs=total_epochs, dropout=dropout[1]).to(device)
        
        []

    return []

def load_parameters_from_txt(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    parameters = {}
    for line in lines:
        key, value = line.strip().split(": ")
        if key in ["dims", "dropout"]:
            parameters[key] = [*map(int, value.split(', '))]
            if key in ["dropout"]:
                parameters[key][1] /= 10
        elif key in ["AE_PATH", "MTc_PATH"]:
            if value == "None":
                parameters[key] = None
            else:
                parameters[key] = value
        else:
            parameters[key] = int(value) if '.' not in value else float(value)

    return parameters