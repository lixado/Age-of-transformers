import os
from DeepRTS import Engine, Constants

workingDir = os.getcwd()

print(workingDir)
print(os.path.dirname(workingDir))

print(Constants.PlayerState.Victory)