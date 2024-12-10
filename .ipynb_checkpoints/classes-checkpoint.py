from torch.utils.data import Dataset, DataLoader
import torch

class createChatDataset(Dataset):
    def __init__(self, X, y ,classes,all_words):
        self.nSamples = len(X)
        self.X_data = X  # Raw data ( numpy array or list
        self.y_data = y  # Raw data ( numpy array or list
        self.classes = classes
        self.all_words = all_words
      
    def __getitem__(self, index):
        # We need to Convert X to Tensor with data type float32
        # y to tensot with dtype long is't just labels( for loss_fn,... )
        x_tensor =torch.tensor(self.X_data[index], dtype= torch.float32)
        y_tensor =torch.tensor(self.y_data[index], dtype= torch.long)
        return x_tensor , y_tensor

    def __len__ (self): 
        return self.nSamples
     
        