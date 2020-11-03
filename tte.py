import torch
import torch.nn.functional as F

# a = torch.rand([2, 16, 16])
# a = F.avg_pool2d(a, 4)
# print(a.shape)

img = torch.randn([6, 2, 2])
img[:2, :, :] = 1
img[2: 4, :, :] = 2
img[4: , :, :] = 3
c, h, w = img.size()
g = 2
# img = img.view(g, c//g, h, w).permute(1, 0, 2, 3).reshape(c, h, w)
print(img.view(g, c//g, h, w).permute(1, 0, 2, 3))