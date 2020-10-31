import numpy as np
import matplotlib.pyplot as plt
import torch

t = np.linspace(-50,50,500)
n = np.poly1d([1,3,1,2])
result = torch.from_numpy(n(t))
print(n.order)

# spatial_img = torch.from_numpy(np.random.rand(3,32, 32))
num_context = int(torch.empty(1).uniform_(20,100).item())
print(num_context)
print(result.shape)
mask = result.new_empty(1, 500).bernoulli_(p=num_context / (500))
img_sparse = mask * result
print(img_sparse.shape)

num_context = int(torch.empty(1).uniform_(20,100).item())
mask = result.new_empty(1, 500).bernoulli_(p=num_context / 500)
func_sparse = mask * result
print(func_sparse.shape)


plt.figure()
plt.scatter(t, img_sparse)
plt.show()
# img_sparse = np.array(img_sparse.permute(1,2,0))
#
#
# spatial_img = torch.from_numpy(np.random.rand(3,32, 32))
# num_context = int(torch.empty(1).uniform_(20,100).item())
# print(num_context)
# mask = spatial_img.new_empty(1, 32, 32).bernoulli_(p=num_context / (32*32))
# img_sparse = mask * spatial_img
# img_sparse = np.array(img_sparse.permute(1,2,0))
#
# print(num_context)
# print(mask.shape)
# plt.figure()
# plt.imshow(img_sparse)
# plt.show()