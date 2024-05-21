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
        self.__conv0 = Convolutional(filters_in=in_channel, filters_out=in_channel//2, kernel_size=1, stride=1, pad=0, norm="bn",
                                    activate="leaky")
        #self.__conv0 = Convolutional(filters_in=in_channel, filters_out=in_channel//2, kernel_size=1, stride=1, pad=0)



        self.__conv1 = Convolutional(filters_in=in_channel*2, filters_out=in_channel, kernel_size=1, stride=1, pad=0, norm='bn',
                                    activate="leaky")
        


    def forward(self, x):  # large, medium, small
        x = self.__conv0(x)

        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4), dim=1)

        out = self.__conv1(out)
        return out #
