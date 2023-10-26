import torch
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping():
    def __init__(self, patience=4, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
    
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
            
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        
        self.status = f"{self.counter}/{self.patience}"
        return False

class ModelTrainer:
    
    def __init__(self, model, lr, epochs, device, train_loader_len=None, save_dir="results"):
        
        # Sets model to GPU and basic loss function and optimizer used
        self.device = device
        self.epochs = epochs
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=epochs)
        
        # Place to store metrics of our model throughout training and testing
        self.Metrics = {"TL":[], "VL":[], # T = Training | L = Loss | LA = Average Loss 
                        "TLA":[], "VLA":[], # A = Accuracy | AA = Average Acurracy | TEA = Testing Acurracy
                        "TA":[], "VA":[],
                        "TAA":[], "VAA":[],
                        "TEL":0, "TEA": 0} 
        
        self.train_len = None
        self.val_len = None
        self.save_dir = save_dir
        self.writer = SummaryWriter(save_dir)
    
    def train_loop(self, progress_bar, loss_fn, epoch):
        
            # Training loop
            for batch in progress_bar:

                loss, acc = self.model.training_step(batch, self.optimizer, loss_fn, epoch)
                # Update tqdm progress bar description with current loss
                progress_bar.set_description(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss:.4f}")
                
                # Append the current loss to the loss_values list
                self.Metrics["TL"].append(loss)
                self.Metrics["TA"].append(acc)

            self.Metrics["TLA"].append(np.average(self.Metrics["TL"][-self.train_len:]))
            self.Metrics["TAA"].append(np.average(self.Metrics["TA"][-self.train_len:]))

    def validation_loop(self, progress_bar, loss_fn, epoch):

        for batch in progress_bar:

            val_loss, val_acc = self.model.validation_step(batch, loss_fn)
                       
            # Update tqdm progress bar description with current loss
            progress_bar.set_description(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {val_loss:.4f}")

            self.Metrics["VL"].append(val_loss)
            self.Metrics["VA"].append(val_acc)
        
        self.Metrics["VLA"].append(np.average(self.Metrics["VL"][-self.val_len:]))
        self.Metrics["VAA"].append(np.average(self.Metrics["VA"][-self.val_len:]))

    def fit(self, train_dataloader, val_dataloader, loss_fn):
        
        self.train_len = train_dataloader.__len__()
        self.val_len = val_dataloader.__len__()
        
        
        # Start the training loop
        best_loss = 1000
        for epoch in range(self.epochs):
            
            # Training loop
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            self.train_loop(progress_bar, loss_fn, epoch)
        
            # Print the average loss for this epoch
            print(f"Epoch [{epoch + 1}/{self.epochs}], \
                  Training Loss: {self.Metrics['TLA'][-1]} | \
                  Training Accuracy: {self.Metrics['TAA'][-1]}")

            # Validation loop
            progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            self.validation_loop(progress_bar, loss_fn, epoch)

            # Print the average validation loss for this epoch
            print(f"Epoch [{epoch + 1}/{self.epochs}], \
                  Validation Loss: {self.Metrics['VLA'][-1]} | \
                  Validation Accuracy: {self.Metrics['VAA'][-1]}")
            
            if self.Metrics['VLA'][-1] < best_loss:
                temp_dir = self.save_dir + f"/{str(epoch)}.pt"
                best_loss = self.Metrics['VLA'][-1]
                torch.save(self.model.state_dict(), temp_dir)

            
            self.writer.add_scalar('training_loss', self.Metrics['TLA'][-1], epoch)
            self.writer.add_scalar('validation_loss', self.Metrics['VLA'][-1], epoch)
            
    
    def test(self, test_dataloader, loss_fn):
        # Initialize the tqdm progress bar

        running_loss = 0.0
        running_acc = 0.0

        # Training loop
        for batch in tqdm(test_dataloader):

            loss, acc = self.model.validation_step(batch, self.optimizer, loss_fn)

            running_loss += loss
            running_acc += acc

        test_loss_avg = running_loss / test_dataloader.__len__()
        test_acc_avg = running_acc / test_dataloader.__len__()

        self.Metrics["TEL"] = test_loss_avg
        self.Metrics["TEA"] = test_acc_avg

        print(f"Validation Loss: {self.Metrics['TEL']} | \
                Validation Accuracy: {self.Metrics['TEA']}")
    
    
    def display_loss(self):
        ###### Plotting Metrics on Training ######
        plt.plot(self.Metrics["TLA"][:])
        plt.plot(self.Metrics["VLA"][:])
        plt.plot(loss_validation)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')

        # To print out and show the average over the last EPOCH
        print(np.median(self.Metrics["TL"][-1078:]))
        print(np.median(self.Metrics["VL"][-308:]))
        plt.show()

    
    def reset(self):
        
        self.Metrics = {"Training Loss":[], "Validation Loss":[], 
                        "Training Accuracy":[], "Validation Accuracy":[],
                        "Test Accuracy":0}
        