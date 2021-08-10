import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List
from ...roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu, points_in_mult_boxes_gpu


class TorchGrouper(nn.Module):
    def __init__(self, min_range, max_range):
        super(TorchGrouper, self).__init__()
        self.max_range = max_range
        range_tensor_positive = torch.arange(min_range, max_range)
        range_tensor_negative = torch.arange(-max_range, -min_range)
        range_tensor = torch.cat((range_tensor_negative, range_tensor_positive))
        per_dim_num = range_tensor.shape[0]
        range_tensor_x = range_tensor.view(1, 1, -1)
        range_tensor_y = range_tensor.view(1, -1, 1)
        range_tensor_z = range_tensor.view(-1, 1, 1)
        range_tensor_x = range_tensor_x.expand(per_dim_num, per_dim_num, per_dim_num)
        range_tensor_y = range_tensor_y.expand(per_dim_num, per_dim_num, per_dim_num)
        range_tensor_z = range_tensor_z.expand(per_dim_num, per_dim_num, per_dim_num)
        index_offset = torch.stack(
            [torch.zeros_like(range_tensor_x), range_tensor_x, range_tensor_y, range_tensor_z], dim=3).type(
            torch.long).cuda()
        self.index_offset = index_offset.view(1, -1, 4).contiguous()
        self.offset_num = self.index_offset.shape[1]

    def forward(self, voxel_maps, grid_positions, features):
        assert len(grid_positions.shape) == 2
        grid_number = grid_positions.shape[0]
        assert grid_positions.shape[1] == 4
        assert len(voxel_maps.shape) == 4
        N, Z, Y, X = voxel_maps.shape
        grid_positions_with_offset_float = grid_positions.view(grid_number, 1, 4) + self.index_offset
        grid_positions_with_offset = grid_positions_with_offset_float.view(-1, 4).type(torch.long)
        grid_positions_with_offset[:, 1].clamp_(min=0, max=Z - 1)
        grid_positions_with_offset[:, 2].clamp_(min=0, max=Y - 1)
        grid_positions_with_offset[:, 3].clamp_(min=0, max=X - 1)
        sampled_idx = voxel_maps[
            grid_positions_with_offset[:, 0], grid_positions_with_offset[:, 1], grid_positions_with_offset[:,
                                                                                2], grid_positions_with_offset[:, 3]]
        features = torch.cat((torch.zeros(1, features.shape[1]).type(features.type()), features), dim=0)
        features = torch.transpose(features, 0, 1)
        sampled_features = features[:, (sampled_idx + 1).long()]
        sampled_features = sampled_features.view(1, features.shape[0], grid_number, self.offset_num)
        grid_positions_feature = grid_positions_with_offset_float.view(grid_number, self.offset_num, 4)[:, :,
                                 1:].unsqueeze(
            dim=0)
        grid_positions_feature = grid_positions_feature.permute(0, 3, 1, 2)
        grid_positions_feature = grid_positions_feature - grid_positions_feature.long().float()
        grid_positions_feature = grid_positions_feature + torch.transpose(self.index_offset[0][:, :3], 0, 1).view(1, 3,
                                                                                                                  1,
                                                                                                                  self.offset_num)
        empty_mask = (sampled_idx + 1).view(grid_number, self.offset_num).sum(dim=1) == 0
        return sampled_features, grid_positions_feature, empty_mask


class NeighborVoxelSAModuleMSG(nn.Module):
    def __init__(self, *, query_ranges: List[List[int]], radii: List[float],
                 nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method='max_pool',
                 grouper_version="v1", disable_xyz=False):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)
        self.grouper_version = grouper_version
        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        self.disable_xyz = disable_xyz
        if self.disable_xyz:
            print("Masking out xyz infos, may cause performance drops")

        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            if grouper_version == 'v1':
                self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            else:
                self.groupers.append(TorchGrouper(radius, max_range))

            mlp_spec = mlps[i]

            cur_mlp_in = nn.Sequential(
                nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[1])
            )

            cur_mlp_pos = nn.Sequential(
                nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp_spec[1])
            )

            cur_mlp_out = nn.Sequential(
                nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[2]),
                nn.ReLU()
            )

            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, \
                new_coords, features, voxel2point_indices, rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        # change the order to [batch_idx, z, y, x]
        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        if self.grouper_version == 'v1':
            new_coords = new_coords.int()
        elif self.grouper_version == 'v2':
            pass
        else:
            raise NotImplementedError
        grouped_xyz_list = []

        new_features_list = []
        for k in range(len(self.groupers)):
            # features_in: (1, C, M1+M2)
            features_in = features.permute(1, 0).unsqueeze(0)
            features_in = self.mlps_in[k](features_in)
            # features_in: (1, M1+M2, C)
            features_in = features_in.permute(0, 2, 1).contiguous()
            # features_in: (M1+M2, C)
            features_in = features_in.view(-1, features_in.shape[-1])
            # grouped_features: (M1+M2, C, nsample)
            # grouped_xyz: (M1+M2, 3, nsample)
            if self.grouper_version == 'v1':
                grouped_features, grouped_xyz_src, empty_ball_mask = self.groupers[k](
                    new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices
                )
                grouped_features[empty_ball_mask] = 0
                # grouped_features: (1, C, M1+M2, nsample)
                grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
                # grouped_xyz: (M1+M2, 3, nsample)
                grouped_xyz = grouped_xyz_src - new_xyz.unsqueeze(-1)
                grouped_xyz[empty_ball_mask] = 0
                # grouped_xyz: (1, 3, M1+M2, nsample)
                grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
                if rois is not None:
                    # M1 + M2, 7
                    rois = rois.clone().view(rois.shape[0] * rois.shape[1], 1, rois.shape[2])
                    roi_num = rois.shape[0]
                    # grouped_xyz_src_transpose: M1+M2, nsample, 3
                    M, C, NSAMPLE = grouped_xyz_src.shape
                    grouped_xyz_src_transpose = grouped_xyz_src.permute(0, 2, 1).reshape(roi_num,
                                                                                         M * NSAMPLE // roi_num, C)
                    batched_mask = points_in_boxes_gpu(grouped_xyz_src_transpose, rois)
                    inside_points = batched_mask == 0
                    inside_points_mask = inside_points.view(1, 1, M, NSAMPLE)
                    grouped_xyz = grouped_xyz * inside_points_mask
                    grouped_features = grouped_features * inside_points_mask

            elif self.grouper_version == 'v2':
                grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](voxel2point_indices, new_coords,
                                                                                  features_in)
            else:
                raise NotImplementedError
            grouped_xyz_list.append(grouped_xyz_src)
            # grouped_xyz: (1, C, M1+M2, nsample)
            position_features = self.mlps_pos[k](grouped_xyz)
            if self.disable_xyz:
                new_features = grouped_features + position_features * 0
            else:
                new_features = grouped_features + position_features
            new_features = self.relu(new_features)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'bilinear':
                distance = torch.pow(grouped_xyz, 2).sum(dim=1, keepdim=True)
                similarity = 1. / (distance + 1e-8)
                similarity = similarity / similarity.sum(dim=-1, keepdim=True)
                new_features = new_features * similarity
                new_features = torch.sum(new_features, dim=-1)
            else:
                raise NotImplementedError

            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        # (M1 + M2 ..., C)
        new_features = torch.cat(new_features_list, dim=1)
        # return new_features, grouped_xyz_list
        return new_features
