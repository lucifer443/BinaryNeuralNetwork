import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier
from mmcls.models.backbones.binary_utils.binary_convs import IRConv2d, RAConv2d


def all_3x3_kernel():
    res = []
    for i in range(512):
        bl = np.array(list(map(int, bin(i)[2:].zfill(9)))).reshape(3, 3)
        res.append(bl)
    pos = np.stack(res)
    neg = pos - 1
    res = pos | neg
    return res.astype(np.float32)

ALL_3x3_KERNEL = torch.from_numpy(all_3x3_kernel())

@CLASSIFIERS.register_module()
class BinaryClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, ce_decay=0.):
        super(BinaryClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.entropy_decay = ce_decay

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(BinaryClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        
        ce = self.entropy_loss()
        losses['CE'] = ce
        if self.entropy_decay > 0.:
            losses['CEloss'] = ce * self.entropy_decay

        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
    
    def entropy_loss(self):
        entropy = torch.tensor([0.]).cuda()
        for m in self.modules():
#             if isinstance(m, (IRConv2d, RAConv2d)):
            if isinstance(m, (IRConv2d, RAConv2d)) and m.out_channels < 256:
#                 print(m.binary_weight)
                entropy += self.compute_entropy(m.binary_weight)
        return entropy
    
    @staticmethod
    def compute_entropy(bw):
        cout, cin = bw.shape[:2]
        all_kernel = torch.from_numpy(all_3x3_kernel())[None, None, :, :, :].expand(cout, cin, 512, 3, 3).cuda()
        bw = bw[:, :, None, :, :].expand_as(all_kernel)
        hist = torch.sum(torch.clamp_min(1 - torch.sum(torch.abs(bw - all_kernel), dim=[3, 4]), 0), dim=[0, 1])
    #     print(hist)
        p = hist / (cout * cin)
        entropy = -1/512 * torch.log2(torch.clamp(p, 1e-8, 1-1e-8))
        return entropy.mean()
    
#     @staticmethod
#     def compute_entropy(data):
#         cout, cin, k, _ = data.shape
#         if k != 3:
#             raise ValueError("Compute entropy only support 3x3 kernel")
#         binary_mask = torch.pow(2.0, torch.arange(k**2, dtype=torch.float).cuda()).reshape(k, k)
# #         int_w = torch.sum(binary_mask.expand_as(kernel) * data.clamp_min(0), dim=[2, 3])
#         int_bw = torch.sum(binary_mask.expand_as(data) * torch.relu(data), dim=[2, 3])
#         p = torch.sum(torch.clamp_min(1 - torch.abs(int_bw - torch.arange(512).cuda().reshape((512, 1, 1)).expand(512, cout, cin)), 0), dim=[1, 2]) / (cout * cin) # compute probability
#         entropy = -1/512 * torch.log2(torch.clamp(p, 1e-8, 1-1e-8))
#         return entropy.mean()

    
