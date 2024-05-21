import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv_module import Convolutional

class SPP(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, in_channel):
        super(SPP, self).__init__()
        self.__conv0 = Convolutional(filters_in=in_channel, filters_out=in_channel*2, kernel_size=1, stride=1, pad=0, norm="bn",
                                    activate="leaky")
        #self.__conv0 = Convolutional(filters_in=in_channel, filters_out=in_channel//2, kernel_size=1, stride=1, pad=0)



        self.__conv1 = Convolutional(filters_in=in_channel*2, filters_out=in_channel, kernel_size=1, stride=1, pad=0, norm='bn',
                                    activate="leaky")
        


    def forward(self, x):  # large, medium, small
        out = self.__conv0(x)

        out = self.__conv1(out)
        return out #
