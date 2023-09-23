
import torch

n = 2
m = 2
batch_size = 10
M = torch.randn((n, m))
v = torch.randn((batch_size -1 , batch_size, m))

# MT = M.T
Mv = v @ M

print(Mv.shape)