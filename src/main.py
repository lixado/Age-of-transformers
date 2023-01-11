import torch
import matplotlib
import time

from logger import Logger

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())

print(time.strftime("%Y%m%d-%H-%M-%S"))

logger = Logger()
