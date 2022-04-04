import os, json
from torch.utils.data import Dataset
import torch

class Cmip6(Dataset):
    def __init__(self, dataset_json):

        self.data = None
        with open(dataset_json, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        lat = torch.tensor(self.data[idx]['lat']).float()
        long = torch.tensor(self.data[idx]['long']).float()
        time = torch.tensor(self.data[idx]['time']).float()
        temp = torch.tensor(self.data[idx]['temp']).float()

        sample = {'lat': lat, 'long': long, 'time': time, 'temp': temp}

        return sample
