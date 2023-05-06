import os
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import torch
import itertools
from transformers import DecisionTransformerModel, DecisionTransformerConfig
from functions import sample_with_order
import numpy as n


class DecisionTransformer_Agent:
    def __init__(self, state_dim, action_space_dim, device, max_steps, batch_size):
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.device = device

        self.state_dim_flatten = np.prod(state_dim)

        self.max_ep_length = 2**11#max_steps # maximum number that can exists in timesteps
        self.n_positions = 2**10 # The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        
        
        print("max_ep_length: ", self.max_ep_length)
        config = DecisionTransformerConfig(self.state_dim_flatten, action_space_dim, max_ep_len=self.max_ep_length, n_positions=self.n_positions, action_tanh=True)
        self.net = DecisionTransformerModel(config).to(device=self.device)
        self.batch_size = batch_size


        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.01
        self.curr_step = 0

        """
            Memory
        """
        # sequences
        self.max_sequence_length = int(self.n_positions/3) # not sure why / 3 might be [R1, S1, A1, R2, S2, A2]
        print("max_sequence_length: ", self.max_sequence_length)
        self.states_sequence = deque(maxlen=self.max_sequence_length)
        self.actions_sequence = deque(maxlen=self.max_sequence_length)
        self.timesteps_sequence = deque(maxlen=self.max_sequence_length)
        self.rewards_sequence = deque(maxlen=self.max_sequence_length)

        arr = np.zeros(state_dim)
        totalSizeInBytes = (arr.size * arr.itemsize * self.max_sequence_length)
        print(f"Need {(totalSizeInBytes*(1e-9)):.2f} Gb ram for states sequence.")


        param_size = 0
        for param in self.net.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.net.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_model_mb = (param_size + buffer_size) / 1024**2
        size_model_gb = size_model_mb / 1024

        arr = np.zeros(state_dim)
        totalSizeInBytes = (arr.size * arr.itemsize * self.max_sequence_length * self.batch_size)
        print(f"Need {(totalSizeInBytes*(1e-9) + size_model_gb):.2f} Gb Vram for states sequence in learning.")


        #self.save_every = 5e5  # no. of experiences between saving model

        """
            Q learning
        """
        self.gamma = 0.9
        self.learning_rate = 0.0025
        self.learning_rate_decay = 0.999985

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)
        self.loss_fn = lambda s_pred, a_pred, r_pred, s, a, r: torch.mean((a_pred-a)**2)
        self.burnin = 1e3  # min. experiences before training
        #assert( self.burnin >  self.batch_size)
        self.learn_every = 1  # no. of experiences between updates to Q_online
        self.sync_every = 1e6  # no. of experiences between Q_target & Q_online sync

        #self.attention_mask = torch.ones((self.batch_size, 1), device=self.device, dtype=torch.float32)  # if None default is full attention for all nodes (b, t)


    def act(self):
        """
        """

        pred_arr = [None for _ in range(self.action_space_dim)]
        if (random.random() < self.exploration_rate): # EXPLORE
            actionIdx = random.randint(0, self.action_space_dim-1)
        else:
            with torch.no_grad():
                sequence_length = len(self.states_sequence)

                #target_return = torch.tensor(1, device=self.device, dtype=torch.float32).unsqueeze(0) # create extra dim for batch
                states = torch.tensor(np.array(self.states_sequence), device=self.device, dtype=torch.float32).reshape(1, sequence_length, self.state_dim_flatten) # prev states
                actions = torch.tensor(np.array(self.actions_sequence), device=self.device, dtype=torch.float32).reshape(1, sequence_length, self.action_space_dim) # create extra dim for batch
                rewards = torch.tensor(self.rewards_sequence, device=self.device, dtype=torch.float32).reshape(1, sequence_length, 1) # create extra dim for batch # not sure what the other is
                timesteps = torch.tensor(self.timesteps_sequence, device=self.device, dtype=torch.long).reshape(1, sequence_length) # create extra dim for batch
                attention_mask = torch.ones((1, sequence_length), device=self.device, dtype=torch.float32) # if None default is full attention for all nodes (b, t)


                state_preds, action_preds, return_preds = self.net(states=states,
                    actions=actions,
                    rewards=None, #not used in foward pass https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L831
                    returns_to_go=rewards,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=False)

                # remove batch dim
                state_preds, action_preds, return_preds = torch.squeeze(state_preds, 0), torch.squeeze(action_preds, 0), torch.squeeze(return_preds, 0)

                actionIdx = torch.argmax(action_preds[-1]).item()
                pred_arr = torch.squeeze(action_preds[-1]).detach().cpu().numpy()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actionIdx, pred_arr

    def append(self, state, actionIndex, timestep, reward):

        # save to sequences
        try:
            self.states_sequence.append(state)
            # save action tensor as [0,0,1,0,0,0] depending on which action
            actionArr = np.zeros(self.action_space_dim)
            actionArr[actionIndex] = 1
            self.actions_sequence.append(actionArr)
            self.timesteps_sequence.append(timestep)
            self.rewards_sequence.append(self.rewards_sequence[-1] - reward)
        except:
            print("Need more memory or decrease sequence size in agent.")
            quit()

    def recall(self):
        stateBatch = []
        actionBatch = []
        rewardBatch = []
        timestepsBatch = []

        currentSize = len(self.states_sequence)
        # sample random sequences between (1, self.max_sequence_length)
        k = random.randint(1, currentSize)
        if k >= self.max_sequence_length: # cant use bigger then allowed
            k = random.randint(1, self.max_sequence_length-1)
        for _ in range(self.batch_size):
            k_start = random.randint(0, currentSize-k)

            stateBatch.append(list(itertools.islice(self.states_sequence, k_start, k_start+k)))
            actionBatch.append(list(itertools.islice(self.actions_sequence, k_start, k_start+k)))
            rewardBatch.append(list(itertools.islice(self.rewards_sequence, k_start, k_start+k)))
            timestepsBatch.append(list(itertools.islice(self.timesteps_sequence, k_start, k_start+k)))

        seq_length = len(stateBatch[0])
        stateBatch = torch.tensor(np.array(stateBatch), device=self.device, dtype=torch.float32).reshape(self.batch_size, seq_length, self.state_dim_flatten)
        actionBatch = torch.tensor(np.array(actionBatch), device=self.device, dtype=torch.float32)
        rewardBatch = torch.tensor(rewardBatch, device=self.device, dtype=torch.float32).reshape(self.batch_size, seq_length, 1)
        timestepsBatch = torch.tensor(timestepsBatch, device=self.device, dtype=torch.long)

        return stateBatch, actionBatch, rewardBatch, timestepsBatch

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        #if self.curr_step < self.burnin:
        #    return 0, 0 # None, None

        #if self.curr_step % self.learn_every != 0:
        #    return 0, 0 # None, None
        
        # [batchSize, sequenceSize, state_dim]
        totalLoss = 0
        avgQ = 0
        j = 3
        for _ in range(j):
            s, a, r, t = self.recall()

            attention_mask = torch.ones((s.shape[0], s.shape[1]), device=self.device, dtype=torch.float32)

            # [R1, S1, A1, R2, S2, A2...]


            state_preds, action_preds, return_preds = self.net(states=s,
                    actions=a,
                    rewards=None, #not used in foward pass https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L831
                    returns_to_go=r,
                    timesteps=t,
                    attention_mask=self.attention_mask,
                    return_dict=False)
            
            loss = self.loss_fn(return_preds, r) + self.loss_fn(state_preds, s)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            totalLoss += loss.item()
            avgQ += action_preds.mean().item()
        

        return totalLoss, (avgQ / j)

    def train(self, observations, actions, timesteps, returns_to_go):
        batch_size = len(observations)
        sequence_length = len(observations[0])


        #print("shape: ",observations.shape)
        observations = torch.tensor(observations, device=self.device, dtype=torch.float32).reshape(batch_size, sequence_length, self.state_dim_flatten)


        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        
        rewards_to_go = torch.tensor(returns_to_go, device=self.device, dtype=torch.float32).reshape(batch_size, sequence_length, 1)

        timesteps = torch.tensor(timesteps, device=self.device, dtype=torch.long)
        attention_mask = torch.ones((batch_size, sequence_length), device=self.device, dtype=torch.float32)  # if None default is full attention for all nodes (b, t)

        observation_preds, action_preds, reward_preds = self.net(states=observations,
           actions=actions,
           rewards=None, # not used in foward pass https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L831
           returns_to_go=rewards_to_go,
           timesteps=timesteps,
           attention_mask=attention_mask,
           return_dict=False)

        # remove batch dim
        #state_preds = torch.squeeze(state_preds, 0)
        #action_preds = torch.squeeze(action_preds, 0)
        #return_preds = torch.squeeze(return_preds, 0)

        loss = self.loss_fn(observation_preds, action_preds, reward_preds,
                            observations[:, 1:], actions, rewards_to_go)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item(), 0

    # def train(self, observations, actions, timesteps, rewards):
    #     observations = observations.float().reshape(self.batch_size, 1, self.state_dim_flatten).to(self.device)
    #
    #     actionArr = torch.zeros((self.batch_size, self.action_space_dim))
    #     for i in range(self.batch_size):
    #         ind = actions[i]
    #         actionArr[i][ind] = 1
    #
    #     actions = actionArr.reshape(self.batch_size, 1, self.action_space_dim).to(self.device)
    #
    #     rewards_to_go_cumsum = torch.cumsum(rewards, dim=0)
    #     rewards_to_go = (rewards_to_go_cumsum[-1] - rewards_to_go_cumsum).float().reshape(self.batch_size, 1, 1).to(self.device)
    #     timesteps = timesteps.reshape(self.batch_size, 1).to(self.device)
    #
    #     observation_preds, action_preds, reward_preds = self.net(states=observations,
    #        actions=actions,
    #        returns_to_go=rewards_to_go,
    #        timesteps=timesteps,
    #        attention_mask=self.attention_mask,
    #        return_dict=False)
    #
    #     loss = self.loss_fn(observation_preds, action_preds, reward_preds,
    #                         observations, actions, rewards)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.detach().cpu().item(), 0

    def save(self, save_dir):
        """
            Save the state to directory
        """
        save_path = os.path.join(save_dir, "model.chkpt")
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"Model saved to {save_path}")