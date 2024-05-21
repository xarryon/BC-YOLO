import torch
from torch import nn

class DropBlock(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9):#block越大，0的区域越大； keep_prob 越大，1区域越大
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)

    def calculate_gamma(self, x):
        '''
        (1-p) * (size^2) / (block^2 * (size - block_size + 1)^2)
        '''
        return  (1-self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2) 

    def forward(self, x):

        if (not self.training or self.keep_prob==1): #set keep_prob=1 to turn off dropblock
            return x
        
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
            
        p = torch.ones_like(x) * (self.gamma)

        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                           self.kernel_size,
                                           self.stride,
                                           self.padding)
        
        out =  mask * x * (mask.numel()/mask.sum())

        return out
