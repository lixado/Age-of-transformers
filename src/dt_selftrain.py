import gym
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random

from tqdm import tqdm

from Agents.decisition_transformer import DecisionTransformer_Agent
from Agents.ddqn import DDQN_Agent
from logger import Logger
from functions import CreateVideoFromTempImages, SaveTempImage, NotifyDiscord


class DTDataset(Dataset):
    def __init__(self, games):
        self.games = games             

        self.total_sequences = []
        for game in games:
            this_game_sequences_sizes = []
            if len(game) > 1:
                i = 2
                while i <= len(game): # assuming the whoe game can fit into model
                    this_game_sequences_sizes.append(i)
                    i += 1

                self.total_sequences.append(this_game_sequences_sizes)
            else:
                print("EJBADJADBMBVDM")
        

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

        print("PROBBLEM")


def train_dt_self(config: dict, agent: DecisionTransformer_Agent, gym: gym.Env, logger: Logger):
    save_dir = logger.getSaveFolderPath()

    record_epochs = config["recordEvery"]  # record game every x epochs
    epochs = config["epochs"]
    batch_size = config["batchSize"]
    episodes = config["DTEpisodesGenerate"]
    dt_epochs = config["DTTrainEpochs"]
    epsilon_rate = config["DTEpsilonRate"]
    dtDataMaxSize = config["DTDataMaxSize"]

    best_data = []
    for e in range(epochs):              
        generated_data = []

        # generate data
        agent.net.eval()
        agent.exploration_rate = 1 - (min(e*epsilon_rate, epochs)/epochs)

        for epi in range(episodes):
            real_e = (e*episodes)+(epi)
            record = (real_e % record_epochs == 0) or (e == epochs - 1)  # last part to always record last
            if record:
                print("Recording this epoch")

            observation, info = gym.reset()
            ticks = 0

            done = False
            truncated = False

            actionIndex = -1  # first acction is default Do nothing
            reward = 1200
            agent.states_sequence = [observation]
            agent.rewards_sequence = [reward]
            agent.timesteps_sequence = [ticks]
            actionArr = np.zeros(agent.action_space_dim)
            actionArr[actionIndex] = 1
            agent.actions_sequence = [actionArr]

            memory = []
            while not done and not truncated:
                ticks += 1

                if e == 0: # full random on first epoch
                    agent.exploration_rate = 1
                
                actionIndex, q_values = agent.act()

                gym.save_player_state()

                # Act
                next_observation, reward, done, truncated, info = gym.step(actionIndex)


                agent.append(next_observation, actionIndex, ticks, reward)

                logger.log_step(reward)


                # save data
                action = [0 for _ in range(agent.action_space_dim)]
                action[actionIndex] = 1
                memory.append([observation.flatten(), action, ticks, reward])

                # Record game
                if record:
                    SaveTempImage(logger.getSaveFolderPath(), gym.render(q_values, reward), ticks)

                # Update state
                observation = next_observation


            logger.log_epoch(real_e, agent.exploration_rate, agent.optimizer.param_groups[0]["lr"])
            # Record game
            if record:
                CreateVideoFromTempImages(os.path.join(logger.getSaveFolderPath(), "temp"), (e))
                agent.save(save_dir)
                record = False

            # Fix reward to return to go
            total_reward = sum([step[-1] for step in memory])
            if total_reward > 0:
                for i, step in enumerate(memory):
                    memory[i][-1] = total_reward - memory[i][-1]
                    total_reward = memory[i][-1]
            generated_data.append(memory)


        # add best data
        # save best games before resettin
        rewards_best_data_before = [sum([step[-1] for step in game]) for game in best_data]
        generated_data.sort(key=lambda game: sum([step[-1] for step in game]), reverse=True)
        best_data += generated_data[0:dtDataMaxSize]



        # make sure best games is always maax 40 games
        best_data.sort(key=lambda game: sum([step[-1] for step in game]), reverse=True)

        best_data = best_data[0:dtDataMaxSize]
        rewards_best_data = [sum([step[-1] for step in game]) for game in best_data]
        print(rewards_best_data)

        if rewards_best_data_before == rewards_best_data:
            dtDataMaxSize = int(dtDataMaxSize * (3 / 4))

            if dtDataMaxSize < 2:
                agent.save(save_dir)
                NotifyDiscord(f"Training finished stagnated. Epochs: {epochs} Name: {save_dir}")
                exit()

            print(f"Stagnation: No improvement new size: {dtDataMaxSize}")
            best_data = best_data[0:dtDataMaxSize]


        
        # train agent
        agent.net.train()
        agent.exploration_rate = 0
        dataset = DTDataset(best_data)

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

                small_batch.append(np.array(observations))
                small_batch.append(np.array(actions))
                small_batch.append(np.array(timesteps))
                small_batch.append(np.array(returns_to_go))

                batches.append(small_batch)


            return batches

        trainingLoader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=6) # need to make it so can take multiple batches and sequence lengths should vary

        losses = []
        for _ in tqdm(range(dt_epochs)):
            loss, q = 0.0, 0.0
            for batch in trainingLoader:
                for small_batch in batch:
                    observations = small_batch[0]
                    actions = small_batch[1]
                    timesteps = small_batch[2]
                    returns_to_go = small_batch[3]

                    loss_temp, q_temp = agent.train(observations, actions, timesteps, returns_to_go)
                    loss += loss_temp
                    q += q_temp
                    losses.append(loss_temp)
                    
            logger.log_step_DT(loss, q)

        print(f"Loss avg: {np.mean(losses)} Loss min:  {np.min(losses)} Loss max: {np.max(losses)}")


    # save model
    agent.save(save_dir)
    NotifyDiscord(f"Training finished. Epochs: {epochs} Name: {save_dir}")
    gym.close()