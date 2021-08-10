import torch.nn as nn
from ...utils import common_utils
from .weight_head import WeightHead


class CASHead(nn.Module):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):

        super().__init__()
        self.model_cfg = model_cfg

        self.num_class = num_class
        self.num_stages = len(self.model_cfg.BBOX_HEAD_CONFIGS)

        stage_weights = self.model_cfg.get('STAGE_WEIGHT', None)
        if stage_weights is None:
            stage_weights = [1. for _ in range(self.num_stages)]
        else:
            assert len(stage_weights) == self.num_stages
        self.stage_weights = stage_weights
        model_cfg_list = self.model_cfg.BBOX_HEAD_CONFIGS

        self.box_heads = nn.ModuleList()
        for i in range(self.num_stages):
            single_head = WeightHead(input_channels=input_channels, model_cfg=model_cfg_list[i],
                                     point_cloud_range=point_cloud_range, voxel_size=voxel_size,
                                     num_class=num_class,
                                     **kwargs)
            self.box_heads.append(single_head)

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        batch_cls_preds_list = []
        batch_reg_preds_list = []

        for stage in range(self.num_stages):
            single_head = self.box_heads[stage]
            batch_dict = single_head(batch_dict)
            batch_dict['rois'] = batch_dict['batch_box_preds']
            batch_reg_preds_list.append(batch_dict['batch_box_preds'])
            batch_cls_preds_list.append(batch_dict['batch_cls_preds'])
        if not self.training:
            batch_dict['batch_cls_preds'] = sum(batch_cls_preds_list) / len(batch_cls_preds_list)
        return batch_dict

    def get_loss(self, tb_dict=None):
        loss_value = 0
        for i, head in enumerate(self.box_heads):
            loss_value_, loss_dict = head.get_loss()
            loss_value = loss_value + loss_value_
            tb_dict.update({"stage{}_{}".format(i, k): v for k, v in loss_dict.items()})
        return loss_value, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_heads[0].box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_heads[0].box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
