import torch.nn as nn
import torch

from ....utils.spconv_utils import spconv


class HeightSumCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        x = batch_dict['encoded_spconv_tensor']
        sp_2d = {}
        multi_3d_features = batch_dict['multi_scale_3d_features']
        # for key, feature in multi_3d_features.items():
        #     sp_2d[key] = self.bev_out(feature)
        out = self.bev_out(x)
        batch_dict['spatial_features'] = out
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # batch_dict['spatial_features'] = sp_2d
        return batch_dict


class HeightSumCompressionMulti(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)
        # idx = features_cat.new_ones((features_cat.shape[0], 1))
        # idx_sum = features_cat.new_zeros((indices_unique.shape[0], 1))
        # idx_sum.index_add_(0, _inv, idx)

        # features_unique /= idx_sum
        
        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        x = batch_dict['encoded_spconv_tensor']
        # sp_2d = {}
        # multi_3d_features = batch_dict['multi_scale_3d_features']
        # /
        out = self.bev_out(x)
        batch_dict['encoded_spconv_tensor'] = out
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # batch_dict['multi_spatial_features'] = sp_2d
        # del batch_dict['multi_scale_3d_features']
        return batch_dict