import os
import numpy as np
from torch.utils.data import Dataset
import pickle

class Simple1v1Dataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        data = os.listdir(data_path)
        self.game_paths = [game for game in data if ".pickle" in game]

    def __len__(self):
        return len(self.game_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.path, self.game_paths[idx])
        file = open(path, "rb")
        item = pickle.load(file)
        file.close()
        #for i in item:
            #i[0] = np.array(i[0])
        return item


class DTDataset(Dataset):
    def __init__(self, data):
        self.games = data

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.games[idx]