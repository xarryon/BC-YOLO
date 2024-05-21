import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import config.yolov3_config_voc as cfg


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss


class YoloV3Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5, tunning=False):
        super(YoloV3Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides
        self.tunning = tunning

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, p_base=None, p_d_base=None):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides

        loss_s, loss_s_giou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        loss_m, loss_m_giou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, strides[1])
        loss_l, loss_l_giou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox,
                                                               lbboxes, strides[2])

        loss = loss_l + loss_m + loss_s
        loss_giou = loss_s_giou + loss_m_giou + loss_l_giou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        if self.tunning == True:
            fs_loss_s = self.__cal_fs_loss_per_layer(p[0], p_d[0], p_base[0], p_d_base[0])
            fs_loss_m = self.__cal_fs_loss_per_layer(p[1], p_d[1], p_base[1], p_d_base[1])
            fs_loss_l = self.__cal_fs_loss_per_layer(p[2], p_d[2], p_base[2], p_d_base[2])
            fs_loss = fs_loss_s + fs_loss_m + fs_loss_l
        
            loss = loss + 0.1*fs_loss

        return loss, loss_giou, loss_conf, loss_cls

    def __cal_fs_loss_per_layer(self, p, p_d, p_base, p_d_base):
        #
        # bs, anchor, size, size
        #

        #split 1
        base_class = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]
        #split 2
        #base_class = [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19]
        #split 3
        #base_class = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 18, 19]

        batch_size = p.shape[0]
        L1 = nn.L1Loss(reduction='none')
        #KL = nn.KLDivLoss(reduction='none')
         
        p_ = p[...,5:]
        p_base_ = p_base[...,5:]
        
        base_p = p_[..., base_class]
        base_p_fs = p_[..., base_class]
        
        #fs_cls_mean = torch.mean(torch.max(p[..., 5:], dim=-1)[0], dim=(1,2,3))
        #bs_cls_mean = torch.mean(torch.max(p_base[..., 5:], dim=-1)[0], dim=(1,2,3))

        #fs_conf_var = torch.var(p[..., 4:5], dim=(1,2,3))
        #bs_conf_var = torch.var(p_base[..., 4:5], dim=(1,2,3))

        #fs_conf_mean = nn.Sigmoid()(torch.mean(p[..., 4:5], dim=(1,2,3)))
        #bs_conf_mean = nn.Sigmoid()(torch.mean(p_base[..., 4:5], dim=(1,2,3)))

            #fs_conf_loss = torch.sum(L1(fs_conf, bs_conf))/batch_size
        fs_cls_loss = torch.sum(L1(base_p_fs, base_p))/batch_size

        #fs_cls_loss = torch.sum(KL(torch.log(base_p_fs), base_p))/batch_size
        
        #fs_loss_mean = torch.sum(torch.sum(L1(fs_conf_mean, bs_conf_mean), dim = 1))/batch_size
        #fs_loss_var = torch.sum(L1(fs_conf_var, bs_conf_var))/batch_size

            #fs_loss = fs_conf_loss + fs_cls_loss
            #fs_loss = fs_conf_loss
        fs_loss = fs_cls_loss
        return fs_loss


    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4]

        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]


        # loss giou
        giou = tools.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_giou = label_obj_mask * bbox_loss_scale * (1.0 - giou) * label_mix


        # loss confidence
        iou = tools.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix


        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix


        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls

        return loss, loss_giou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.yolov3 import Yolov3
    net = Yolov3()

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3,  52, 52, 3,26)
    label_mbbox = torch.rand(3,  26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3,26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(cfg.MODEL["ANCHORS"], cfg.MODEL["STRIDES"])(p, p_d, label_sbbox,
                                    label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
