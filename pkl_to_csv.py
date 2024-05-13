import os
import importlib
import torch
import numpy as np
import sys

data = np.load('/home/shaktis/Documents/OmniIsaacGymUR16eEnv/omniisaacgyms/dataset.pkl', allow_pickle=True)

print(data.shape)

print(data[0])

np.savetxt("18-35-30.csv", data)