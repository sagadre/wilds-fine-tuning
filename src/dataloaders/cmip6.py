import os, json
from torch.utils.data import Dataset

class Cmip6(Dataset):
    def __init__(self, dataset_json):

        self.dataset = None
        with open(dataset_json, 'r') as f:
            self.dataset = json.load(f)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        lat = self.data[idx]['lat']
        long = self.data[idx]['long']
        time = self.data[idx]['time']
        temp = self.data[idx]['temp']

        sample = {'lat': lat, 'long': long, 'time': time, 'temp': temp}

        return sample
