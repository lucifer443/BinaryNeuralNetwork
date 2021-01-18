import torch
from torch.nn import functional as F
from torch.nn.modules import loss

from mmcv.runner import load_checkpoint
from mmcv import Config

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck, build_classifier
from .image import ImageClassifier


@CLASSIFIERS.register_module()
class DistillingImageClassifier(ImageClassifier):

    def __init__(self, backbone, distill, neck=None, head=None, pretrained=None):
        super(DistillingImageClassifier, self).__init__(backbone, neck, head, pretrained)
        teacher_config = Config.fromfile(distill.teacher_cfg)
        self.teacher_model = build_classifier(teacher_config.model)
        load_checkpoint(self.teacher_model, distill.teacher_ckpt)
        self.distill_loss_weight = distill.loss_weight
        self.kd_loss = DistributionLoss()
        self.only_kd = distill.only_kdloss
        
        # freeze teacher
        for m in self.teacher_model.modules():
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        super(DistillingImageClassifier, self).init_weights(pretrained)

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
        teacher_x = self.teacher_model.extract_feat(img)
        model_output = self.head.fc(x)
        real_output = self.teacher_model.head.fc(teacher_x)

        losses = dict()
        if not self.only_kd:
            loss = self.head.forward_train(x, gt_label)
            losses.update(loss)
        losses['KD_loss'] = self.kd_loss(model_output, real_output)

        return losses

class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss