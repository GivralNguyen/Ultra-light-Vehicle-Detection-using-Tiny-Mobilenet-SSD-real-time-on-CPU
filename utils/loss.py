import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import box_processing as box_utils

class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 0.25):
        """
            focusing is parameter that can adjust the rate at which easy
            examples are down-weighted.
            alpha may be set by inverse class frequency or treated as a hyper-param
            If you don't want to balance factor, set alpha to 1
            If you don't want to focusing factor, set gamma to 1 
            which is same as normal cross entropy loss
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, conf_preds, loc_preds, conf_targets, loc_targets):
        """
            Args:
                predictions (tuple): (conf_preds, loc_preds)
                    conf_preds shape: [batch, n_anchors, num_cls]
                    loc_preds shape: [batch, n_anchors, 4]
                targets (tensor): (conf_targets, loc_targets)
                    conf_targets shape: [batch, n_anchors]
                    loc_targets shape: [batch, n_anchors, 4]
        """

        ############### Confiden Loss part ###############
        """
        #focal loss implementation(1)
        pos_cls = conf_targets > -1 # exclude ignored anchors
        mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
        conf_t = conf_targets[pos_cls].view(-1).clone()
        p = F.softmax(conf_p, 1)
        p = p.clamp(1e-7, 1. - 1e-7) # to avoid loss going to inf
        c_mask = conf_p.data.new(conf_p.size(0), conf_p.size(1)).fill_(0)
        c_mask = Variable(c_mask)
        ids = conf_t.view(-1, 1)
        c_mask.scatter_(1, ids, 1.)
        p_t = (p*c_mask).sum(1).view(-1, 1)
        p_t_log = p_t.log()
        # This is focal loss presented in ther paper eq(5)
        conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)
        conf_loss = conf_loss.sum()
        """
    
        # focal loss implementation(2)
        pos_cls = conf_targets >-1
        mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
        conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
        p_t_log = -F.cross_entropy(conf_p, conf_targets[pos_cls], reduction='sum')
        p_t = torch.exp(p_t_log)

        # This is focal loss presented in the paper eq(5)
        conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)

        ############# Localization Loss part ##############
        pos = conf_targets > 0 # ignore background
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        loc_p = loc_preds[pos_idx].view(-1, 4)
        loc_t = loc_targets[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        num_pos = pos.long().sum(1, keepdim = True)
        N = max(num_pos.data.sum(), 1) # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes 
        conf_loss /= N # exclude number of background?
        loc_loss /= N

        return loc_loss, conf_loss

    def one_hot(self, x, n):
        y = torch.eye(n)
        return y[x]