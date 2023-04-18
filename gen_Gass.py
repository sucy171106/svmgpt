import matplotlib.pyplot as plt
import torch

mean = torch.zeros((2,))
Id = torch.eye(2)*0.05  # Identity matrix for multivariate normal std
norm = torch.distributions.multivariate_normal.MultivariateNormal(mean, Id)
X = norm.sample((10,))  # X data
X2 = norm.sample((10,))  # X test