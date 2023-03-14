import os

from torch.utils.data import Dataset
import pickle

class AllActions1v1Dataset(Dataset):
    def __init__(self, data_path):
        self.game_paths = os.listdir(data_path)

    def __len__(self):
        return len(self.game_paths)

    def __getitem__(self, idx):
        file = open(self.game_paths[idx], "rb")
        item = pickle.load(file)
        file.close()
        return item

def pad_batch():
    pass