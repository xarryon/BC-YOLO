import torch.nn as nn
import torch
import math
from ..layers.conv_module import Convolutional, DeformableConvolutional
from .dropblock import AttentiveDropBlock


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride, in_channel, out_channel_cls, out_channel_det):
        super(Yolo_head, self).__init__()


        self.__conv_0 = Convolutional(filters_in=in_channel, filters_out=in_channel*2, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")

        self.__conv_cls = Convolutional(filters_in=in_channel*2, filters_out=out_channel_cls, kernel_size=1,
                                       stride=1, pad=0)

        self.__conv_det = Convolutional(filters_in=in_channel*2, filters_out=out_channel_det, kernel_size=1,
                                       stride=1, pad=0)


        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride
        
        self.dropblock = AttentiveDropBlock(block_size=7)
        
        #self.__init_cls_weight()

    def forward(self, r, drop = False):
        if drop == True:
            r = self.dropblock(r)

        out = self.__conv_0(r)
        #out = self.dropblock(out)

        p_cls = self.__conv_cls(out)
        p_det = self.__conv_det(out)

        p = torch.cat((p_det, p_cls), dim = 1)

        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)

        p_de = self.__decode(p.clone())

        return (p, p_de)

    def __init_cls_weight(self):
        prior = 0.01
        self.__conv_cls.__conv.weight.data.fill_(0)
        self.__conv_cls.__conv.bias.data.fill_(-math.log((1.0 - prior)/prior))
    
    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
