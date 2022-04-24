import os, json
from torch.utils.data import Dataset
import torch
import math

class Cmip6(Dataset):
    def __init__(self, dataset_json):

        self.data = None
        with open(dataset_json, 'r') as f:
            self.data = json.load(f)
        self.temp_mean_kelvin = 277.3833287606154
        self.temp_std_kelvin = 21.186756402995293

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # torch.tensor(self.data[idx]['temp'])
        lat = torch.tensor(self.data[idx]['lat']/90).float()
        long = torch.tensor(math.cos(self.data[idx]['long'])).float()
        time = torch.tensor((self.data[idx]['time'] - 1850) / (2020 - 1850)).float()
        temp = torch.tensor((self.data[idx]['temp'] - self.temp_mean_kelvin)/self.temp_std_kelvin).float()

        sample = {'lat': lat, 'long': long, 'time': time, 'temp': temp}

        return sample

def pred_to_interpretable(pred):
    return (pred * 21.186756402995293) + 277.3833287606154