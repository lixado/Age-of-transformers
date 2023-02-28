# Algorithm destilation https://arxiv.org/abs/2210.14215


"""
import torch
from transformers import DecisionTransformerModel, DecisionTransformerConfig


modelConfig = DecisionTransformerConfig() # https://huggingface.co/docs/transformers/model_doc/decision_transformer

model = DecisionTransformerModel(modelConfig) 
"""

from transformers import DecisionTransformerModel
import torch
import gym

model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
# evaluation
device = "cpu"
model = model.to(device)
model.eval()

env = gym.make("Hopper-v3")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

state = env.reset()
states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
target_return = torch.zeros(1, 1, device=device, dtype=torch.float32)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

# forward pass
with torch.no_grad():
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=target_return,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False)
