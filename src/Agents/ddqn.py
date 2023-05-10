import os
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import torch
import torch._dynamo.config
from torchvision.models import resnet50
from torchvision import transforms

#based on pytorch RL tutorial by yfeng997: https://github.com/yfeng997/MadMario/blob/master/agent.py
class DDQN_Agent:
    def __init__(self, state_dim, action_space_dim, config):
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.device = config["device"]

        self.net = DDQN(self.state_dim, self.action_space_dim).float().to(device=self.device)

        self.exploration_rate = config["exploration_rate"]
        self.exploration_rate_decay = config["exploration_rate_decay"]
        self.exploration_rate_min = config["exploration_rate_min"]
        self.curr_step = 0
        """
            Memory
        """
        self.deque_size = config["deque_size"]
        arr = np.zeros(state_dim)
        totalSizeInBytes = (arr.size * arr.itemsize * 2 * self.deque_size) # *2 because 2 observations are saved
        print(f"Need {(totalSizeInBytes*(1e-9)):.2f} Gb ram")
        self.memory = deque(maxlen=self.deque_size)
        self.batch_size = config["batch_size"]
        print(f"Need {((arr.size * arr.itemsize * 2 * self.batch_size)*(1e-9)):.2f} Gb VRAM")
        #self.save_every = 5e5  # no. of experiences between saving model

        """
            Q learning
        """
        self.learning_rate = config["learning_rate"]

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = config["burnin"]  # min. experiences before training
        assert( self.burnin >  self.batch_size)
        self.learn_every = config["learn_every"]  # no. of experiences between updates to Q_online
        self.sync_every = config["sync_every"]  # no. of experiences between Q_target & Q_online sync

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
                state = torch.tensor(state).float().to(device=self.device)
                state = state.unsqueeze(0) # create extra dim for batch

                neuralNetOutput = self.net(state, model="online")
                actionIdx = torch.argmax(neuralNetOutput).item()
                pred_arr = neuralNetOutput[0].detach().cpu().numpy()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actionIdx, pred_arr

    def cache(self, state, next_state, action, reward, done):
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

        try:
            self.memory.append((state, next_state, action, reward, done))
        except:
            print("Need more memory or decrease Deque size in agent.")
            quit()

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
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

        # Get TD Estimate, make predictions for the each memory
        td_est = self.td_estimate(state, action)

        # Get TD Target make predictions for next state of each memory
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)


        return (td_est.mean().item(), loss)

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def td_estimate(self, state, action):
        """
            Output is batch_size number of rewards = Q_online(s,a) * 32
        """
        modelOutPut = self.net(state, model="online")
        current_Q = modelOutPut[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
            Output is batch_size number of Q*(s,a) = r + (1-done) * gamma * Q_target(s', argmax_a'( Q_online(s',a') ) )
        """
        next_state_Q = self.net(next_state, model="online") 
        best_action = torch.argmax(next_state_Q, axis=1) # argmax_a'( Q_online(s',a') ) 
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action] # Q_target(s', argmax_a'( Q_online(s',a') ) )
        return (reward + (1 - done.float()) * self.gamma * next_Q).float() # Q*(s,a)

    def loadModel(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")


    def save(self, save_dir):
        """
            Save the state to directory
        """
        save_path = os.path.join(save_dir, "model.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Model saved to {save_path}")


class DDQN(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        c, h, w = state_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
           p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)