import torch

# 1D input, 1D output
simplest = torch.tensor([[1, 2], [-1, 0], [1, 2]], dtype=torch.double).unsqueeze(2)

# 2D input
multi_input = torch.tensor([[1, 1, 2], [-1, -2, 0], [1, 0.5, 2]], dtype=torch.double).unsqueeze(2)
