import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List


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
        self.offset_num = self.index_offset.shape[0]

    def forward(self, voxel_maps, grid_positions, features):
        assert len(grid_positions.shape) == 2
        grid_number = grid_positions.shape[0]
        assert grid_positions.shape[1] == 4
        assert len(voxel_maps.shape) == 4
        N, Z, Y, X = voxel_maps.shape
        grid_positions_with_offset = grid_positions.view(grid_number, 1, 4) + self.index_offset
        grid_positions_with_offset = grid_positions_with_offset.view(-1, 4).type(torch.long)
        grid_positions_with_offset[:, 1].clamp_(min=0, max=Z - 1)
        grid_positions_with_offset[:, 2].clamp_(min=0, max=Y - 1)
        grid_positions_with_offset[:, 3].clamp_(min=0, max=X - 1)
        sampled_idx = voxel_maps[
            grid_positions_with_offset[:, 0], grid_positions_with_offset[:, 1], grid_positions_with_offset[:,
                                                                                2], grid_positions_with_offset[:, 3]]
        features = torch.cat((torch.zeros(1, features.shape[1]).type(features.type()), features), dim=0)
        features = torch.transpose(features, 0, 1)
        sampled_features = features[:, sampled_idx + 1]
        sampled_features = sampled_features.view(1, features.shape[1], grid_number, self.offset_num)
        grid_positions_feature = grid_positions_with_offset.view(grid_number, self.offset_num, 4)[:, :, 1:3].unsqueeze(
            dim=0)
        grid_positions_feature = grid_positions_feature.permute(0, 3, 1, 2)
        grid_positions_feature = grid_positions_feature - grid_positions_feature.long()
        grid_positions_feature = grid_positions_feature + torch.transpose(self.index_offset[0], 0, 1).view(1, 4, 1,
                                                                                                           self.offset_num)
        empty_mask = (sampled_idx + 1).view(grid_number, self.offset_num).sum(dim=1) == 0
        return sampled_features, grid_positions_feature, empty_mask


class NeighborVoxelSAModuleMSGV2(nn.Module):
    def __init__(self, *, query_ranges: List[List[int]], radii: List[float],
                 nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method='max_pool'):
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

        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
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

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                new_coords, features, voxel2point_indices):
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

            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](voxel2point_indices, new_coords,
                                                                              features_in)

            # grouped_features[empty_ball_mask] = 0
            # grouped_features: (1, C, M1+M2, nsample)
            # grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            # grouped_xyz: (M1+M2, 3, nsample)
            # grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            # grouped_xyz[empty_ball_mask] = 0
            # grouped_xyz: (1, 3, M1+M2, nsample)
            # grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            # grouped_xyz: (1, C, M1+M2, nsample)
            position_features = self.mlps_pos[k](grouped_xyz)
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
            else:
                raise NotImplementedError

            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        # (M1 + M2 ..., C)
        new_features = torch.cat(new_features_list, dim=1)
        return new_features
