import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, classNums, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):
        super(YOLOLoss, self).__init__()
        
        
        
        
        
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask
        self.label_smoothing = label_smoothing
        self.threshold      = 4
        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05
        self.obj_ratio      = 1 * (input_shape[0] * input_shape[1]) / (320 ** 2)
        self.cls_ratio      = 0.5 * (num_classes / 3)
        self.cuda = cuda
        
        self.ClassNums      = classNums
        self.gamma          = 0.99
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
    
    def CBCELoss(self, pred, target):
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        num        = target.size(0)
        
        class_Nums = self.ClassNums.repeat(num,1).cuda()
        ny, _   = (target * class_Nums).max(-1, keepdim=True)
        ny      = ny.float()
        
        balance = (1 - self.gamma) / (1 - torch.pow(self.gamma, ny))
        output  = output * balance
        return output
    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        
        
        
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        
        
        
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half
        
        
        
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area
        
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy),2), axis=-1)
        
        
        
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou             = iou - 1.0*(center_distance)/torch.clamp(enclose_diagonal,min = 1e-6)
        v                = (4 / (math.pi**2)) * torch.pow((torch.atan(b1_wh[...,0]/torch.clamp(b1_wh[...,1],min = 1e-6))-torch.atan(b2_wh[...,0]/torch.clamp(b2_wh[...,1],min = 1e-6))),2)
        alpha            = v/torch.clamp((1.0 - iou + v),min = 1e-6)
        ciou             = ciou - alpha * v
        return ciou
        
    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        
        
        
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        
        
        
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half
        
        
        
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area
        
        
        
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        
        
        
        enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou            = iou - (enclose_area - union_area) / enclose_area
        
        return giou
    
    
    
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    def forward(self, l, inputs, targets=None):
        input, coarse = inputs
        coarse = coarse.permute(0,2,3,1).contiguous()
        coarse = torch.sigmoid(coarse)
        
        
        
        
        
        
        
        
        
        
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        
        
        
        
        
        
        
        
        
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        
        
        
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        
        
        
        
        
        
        
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        
        
        
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        
        
        
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 
        
        
        
        conf = torch.sigmoid(prediction[..., 4])
        
        
        
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        
        
        y_true, noobj_mask = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        
        
        
        
        
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)
        if self.cuda:
            y_true          = y_true.cuda()
            noobj_mask      = noobj_mask.cuda()
        
        loss    = 0
        n       = torch.sum(y_true[..., 4] == 1)
        if n != 0:
            
            
            
            giou        = self.box_ciou(pred_boxes, y_true[..., :4])
            loss_loc    = torch.mean((1 - giou)[y_true[..., 4] == 1])
            loss_cls    = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
            loss        += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            
            
            
            tobj        = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            tobj        = torch.zeros_like(y_true[..., 4])
        loss_conf   = torch.mean(self.BCELoss(conf, tobj))
        
        loss        += loss_conf * self.balance[l] * self.obj_ratio
        
        
        
        
        coarse_true, coarse_trueMask = self.get_coarse(targets, noobj_mask, in_h, in_w)
        coarse_true = coarse_true.cuda()
        
        if len(coarse_true[coarse_trueMask]) != 0:
          loss_coarseTrueCls = torch.mean(self.BCELoss(coarse[coarse_trueMask], coarse_true[coarse_trueMask]))
          loss += loss_coarseTrueCls * self.cls_ratio * 0.5
        return loss
    def get_coarse(self, targets, noobj_mask, in_h, in_w):
      bs = len(targets)
      coarse_true = torch.zeros(bs, in_h, in_w, self.num_classes)
      
      
      for b in range(bs):
        if len(targets[b]) == 0:
          continue
        batch_target = torch.zeros_like(targets[b])
        
        
        
        batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
        batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
        batch_target[:, 4] = targets[b][:, 4]
        batch_target = batch_target.cpu()
        for t, _ in enumerate(batch_target):
          
          
          
          i = torch.floor(batch_target[t, 0]).long()
          j = torch.floor(batch_target[t, 1]).long()
          
          
          
          c = batch_target[t, 4].long()
          coarse_true[b, j, i, c] = 1
        
      temp = torch.sum(coarse_true, dim=-1, keepdim=True)
      coarse_trueMask = temp[...,0]>0
      
      
      
      
      return coarse_true,coarse_trueMask
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]
    def get_target(self, l, targets, anchors, in_h, in_w):
        
        
        
        bs              = len(targets)
        
        
        
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        
        
        
        box_best_ratio = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        
        
        
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):            
            if len(targets[b])==0:
                continue
            batch_target = torch.zeros_like(targets[b])
            
            
            
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            
            
            
            
            
            
            
            
            
            
            
            ratios_of_gt_anchors = torch.unsqueeze(batch_target[:, 2:4], 1) / torch.unsqueeze(torch.FloatTensor(anchors), 0)
            ratios_of_anchors_gt = torch.unsqueeze(torch.FloatTensor(anchors), 0) /  torch.unsqueeze(batch_target[:, 2:4], 1)
            ratios               = torch.cat([ratios_of_gt_anchors, ratios_of_anchors_gt], dim = -1)
            max_ratios, _        = torch.max(ratios, dim = -1)
            for t, ratio in enumerate(max_ratios):
                
                
                
                over_threshold = ratio < self.threshold
                over_threshold[torch.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    
                    
                    
                    i = torch.floor(batch_target[t, 0]).long()
                    j = torch.floor(batch_target[t, 1]).long()
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue
                        if box_best_ratio[b, k, local_j, local_i] != 0:
                            if box_best_ratio[b, k, local_j, local_i] > ratio[mask]:
                                y_true[b, k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        
                        
                        
                        c = batch_target[t, 4].long()
                        
                        
                        
                        noobj_mask[b, k, local_j, local_i] = 0
                        
                        
                        
                        y_true[b, k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[b, k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[b, k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[b, k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[b, k, local_j, local_i, 4] = 1
                        y_true[b, k, local_j, local_i, c + 5] = 1
                        
                        
                        
                        box_best_ratio[b, k, local_j, local_i] = ratio[mask]
                        
        return y_true, noobj_mask
    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        
        
        
        bs = len(targets)
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        
        
        
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)
        
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        
        
        pred_boxes_x    = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        return pred_boxes
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr
    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
