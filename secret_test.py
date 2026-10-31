import torch
import numpy as np
secret=np.array([[1,0,1,1,0,0,1,0,1,0,
                 0,0,1,1,0,0,0,0,1,0,
                 0,0,1,1,0,1,0,0,1,0,
                 0,0,1,1,1,0,0,0,1,1,
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                 ]])
secret = torch.from_numpy(secret).float()

print(secret.shape)
print(secret)