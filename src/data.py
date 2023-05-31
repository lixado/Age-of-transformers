import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
import copy


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
    def __init__(self, games, action_space_dim, data_path=None):
        if data_path is None:
            self.games = copy.deepcopy(games)
        else:
            self.games = []
            data = os.listdir(data_path)
            for path in data:
                game_path = os.path.join(data_path, path)
                f = open(game_path, "rb")
                game = pickle.load(f)
                f.close()

                self.games.append(game)
                for t in game:
                    t[0] = t[0].flatten()
        self.action_space_dim = action_space_dim

        self.total_sequences = []
        for game in self.games:
            this_game_sequences_sizes = []
            if len(game) > 1:
                i = 2
                while i <= len(game):  # assuming the whole game can fit into model
                    this_game_sequences_sizes.append(i)
                    i += 1

                self.total_sequences.append(this_game_sequences_sizes)
            else:
                exit("A game cannot be 1 step")

            # fix action and reward
            for i, step in enumerate(game):
                # translate action to list
                action = [0 for _ in range(self.action_space_dim)]
                actionIndex = step[1]
                action[actionIndex] = 1
                game[i][1] = action

                # return to go
                if i == 0:  # first step
                    total_reward_this_game = sum([step[-1] for step in game])
                    game[0][-1] = total_reward_this_game  # this return to go
                else:
                    game[i][-1] = game[i - 1][-1] - game[i][-1]

    def __len__(self):
        total_length = 0
        for game in self.total_sequences:
            total_length += len(game)

        return total_length

    def __getitem__(self, idx):
        """
            should return a sequence
        """
        x = 0
        for i in range(len(self.total_sequences)):
            for j in range(len(self.total_sequences[i])):
                if x == idx:
                    length = self.total_sequences[i][j]

                    return self.games[i][0:length]
                x += 1


def collate_fn(batch):

    # here sequences will have different sizes and we want the same sizes
    lengths = [len(sequence) for sequence in batch]
    lengths = list(set(lengths))

    batches = []

    for l in lengths:
        small_batch = []

        observations = []
        actions = []
        timesteps = []
        returns_to_go = []
        for sequence in [b for b in batch if len(b) == l]:
            observations.append([step[0] for step in sequence])
            actions.append([step[1] for step in sequence])
            timesteps.append([step[2] for step in sequence])
            returns_to_go.append([step[3] for step in sequence])

        small_batch.append(torch.tensor(np.array(observations), dtype=torch.float32))
        small_batch.append(torch.tensor(np.array(actions), dtype=torch.float32))
        small_batch.append(torch.tensor(np.array(timesteps), dtype=torch.long))
        small_batch.append(torch.tensor(np.array(returns_to_go), dtype=torch.float32).unsqueeze(-1))

        batches.append(small_batch)

    return batches
