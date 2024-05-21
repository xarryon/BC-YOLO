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
        negative correlation to p and block
        '''
        return  (1-self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2) 

    def forward(self, x):

        if (not self.training or self.keep_prob==1): #set keep_prob=1 to turn off dropblock
            return x
        
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x) #gamma越小,产生越少的1,maxpool产生更少的1block,进而通过1-block产生更少的dropblock

        p = torch.ones_like(x) * (self.gamma)

        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                           self.kernel_size,
                                           self.stride,
                                           self.padding)
        
        out =  mask * x * (mask.numel()/mask.sum())
        #print('DB', mask.sum())

        return out


class AttentiveDropBlock(nn.Module):
    def __init__(self, block_size=7, drop_scale = 0.02):  # block越大，0的区域越大； keep_prob 越大，1区域越大
        super(AttentiveDropBlock, self).__init__()
        self.block_size = block_size
        self.drop_scale = drop_scale
        self.keep_prob = 0.9
        self.gamma = None
        #self.gamma_standard = None

        self.chl_avg = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                     nn.Sigmoid())
                                     #nn.Softmax(dim=1))

        self.spl_avg = nn.Sequential(nn.Sigmoid())

        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def calculate_gamma(self, x):
        #print('x',x.shape)
        chl_feat = self.chl_avg(x)
        #print('chl', chl_feat.shape)
        spl_mean = torch.mean(x, dim=1).unsqueeze(1)
        spl_feat = self.spl_avg(spl_mean)
        #print('spl', spl_feat.shape)
        return chl_feat * spl_feat
    
    def calculate_gamma_standard(self, x):
        '''
            (1-p) * (size^2) / (block^2 * (size - block_size + 1)^2)
            negative correlation to p and block
        '''
        return  (1-self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2)

    def forward(self, x):
        if (not self.training):  # set keep_prob=1 to turn off dropblock
            return  x

        #if self.gamma is None:
        self.gamma = self.calculate_gamma(x) * self.drop_scale  # gamma越小,产生越少的1,maxpool产生更少的1block,进而通过1-block产生更少的dropblock
      
        #print(self.gamma.shape)
        #if self.gamma_standard is None:
        #    self.gamma_standard = self.calculate_gamma_standard(x)

        #p = torch.ones_like(x) * (self.gamma)
        
        #print('self.gamma', self.gamma.shape)
        #print('p', p.shape)
        #self.gamma = torch.maximum(p, self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(self.gamma),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        #print('mask', mask.shape) 
        out = mask * x * (mask.numel() / mask.sum())
        #print('ADB', mask.sum())

        return out

if __name__ == '__main__':
    x = torch.randn(32,128,48,48)
    #x = torch.randn(1, 3, 10, 10)
    #db = DropBlock(block_size=7, keep_prob=0.9)
    Attentivedb = AttentiveDropBlock(block_size=7)

    #out = db(x)
    out_2 = Attentivedb(x)

    # tp = torch.randn(1, 1, 3, 3)
    # print(tp)
    # tp = nn.Sigmoid()(tp)
    # print(tp)
