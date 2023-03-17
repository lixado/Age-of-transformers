import os
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import torch
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from functions import sample_with_order


class DecisionTransformer_Agent:
    def __init__(self, state_dim, action_space_dim):
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.state_dim_flatten = state_dim[0]*state_dim[1]*state_dim[2]
        self.max_ep_length = 4096
        config = DecisionTransformerConfig(self.state_dim_flatten, action_space_dim, max_ep_len=self.max_ep_length)
        self.net = DecisionTransformerModel(config).to(device=self.device)

        self.exploration_rate = 1 # 1
        self.exploration_rate_decay = 0.995
        self.exploration_rate_min = 0.0001
        self.curr_step = 0

        """
            Memory
        """
        self.deque_size = 4000
        arr = np.zeros(state_dim)
        totalSizeInBytes = (arr.size * arr.itemsize * 2 * self.deque_size) # * 2 because 2 states are saved in line
        print(f"Need {(totalSizeInBytes*(1e-9)):.2f} Gb ram")
        self.memory = deque(maxlen=self.deque_size)
        self.batch_size = 16
        #self.save_every = 5e5  # no. of experiences between saving model

        """
            Q learning
        """
        self.gamma = 0.9
        self.learning_rate = 0.0025
        self.learning_rate_decay = 0.999985

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e3  # min. experiences before training
        assert( self.burnin >  self.batch_size)
        self.learn_every = 1  # no. of experiences between updates to Q_online
        self.sync_every = 1e6  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
            Given a state, choose an epsilon-greedy action and update value of step.
            Inputs:
            state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        pred_arr = [None for _ in range(self.action_space_dim)]
        if (random.random() < self.exploration_rate): # EXPLORE
            actionIdx = random.randint(0, self.action_space_dim-1)
        else: # EXPLOIT
            with torch.no_grad():
                state = np.array(state)
                #state = state.unsqueeze(0) # create extra dim for batch

                # need to save previous states and actions to send here
                # search in memory up to self.max_ep_length for previous stuff and send in here

                target_return = torch.tensor(1, device=self.device, dtype=torch.float32).reshape(1, 1)
                states = torch.tensor(state, device=self.device, dtype=torch.float32).reshape(1, 1, self.state_dim_flatten) # prev states
                actions = torch.rand((1, self.action_space_dim), device=self.device, dtype=torch.float32) # prev q values
                rewards = torch.rand((1, 1), device=self.device, dtype=torch.float32) # prev rewards
                timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1) # 
                attention_mask = None#torch.zeros(1, 1, device=self.device, dtype=torch.float32) #

                state_preds, action_preds, return_preds = self.net(states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=target_return,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=False)

                actionIdx = torch.argmax(action_preds).item()
                pred_arr = torch.squeeze(action_preds).detach().cpu().numpy()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actionIdx, pred_arr

    def cache(self, state, next_state, action, reward, done, t):
        """
        Store the experience to self.memory (replay buffer)
        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        # not make to np array ans use lazyframes
        state = np.array(state)
        next_state = np.array(next_state)
        state = torch.tensor(state).float()#.to(device=self.device)
        next_state = torch.tensor(next_state).float()#.to(device=self.device)

        action = torch.tensor([action])#.to(device=self.device)
        reward = torch.tensor([reward])#.to(device=self.device)
        done = torch.tensor([done])#.to(device=self.device)
        t = torch.tensor([t])

        try:
            self.memory.append((state, next_state, action, reward, done, t))
        except:
            print("Need more memory or decrease Deque size in agent.")
            quit()


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = sample_with_order(self.memory, self.batch_size)
        state, next_state, action, reward, done, t = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), t.squeeze()

    def learn(self, save_dir):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return 0, 0 # None, None

        if self.curr_step % self.learn_every != 0:
            return 0, 0 # None, None

        # Sample from memory get self.batch_size number of memories
        state, next_state, action, reward, done = self.recall()

        # move everything to gpu here to use less gpu memory but slower training
        state, next_state, action, reward, done = state.to(device=self.device), next_state.to(device=self.device), action.to(device=self.device), reward.to(device=self.device), done.to(device=self.device)



        #return (td_est.mean().item(), loss)
