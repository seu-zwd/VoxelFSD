from functools import partial
import torch
import torch.nn as nn
import numpy as np

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import loss_utils
from ..model_utils import centernet_utils


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, dilation=1):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key, dilation=dilation)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class InceptionAtt(spconv.SparseModule):
    def __init__(self, in_planes, kernel_size=3, stride=1, branch_ratio=0.25, bias=None, norm_fn=None,  indice_key=None):
        super().__init__()
        gc = int(in_planes*branch_ratio)
        k = kernel_size
        self.conv_xy1 = spconv.SubMConv3d(gc, gc, (1, k, k), stride, bias=bias, indice_key=f"{indice_key}_xy1")
        self.conv_xy2 = spconv.SubMConv3d(gc, gc, (1, 5, 5), stride, bias=bias, indice_key=f"{indice_key}_xy2")
        self.conv_xy3 = spconv.SubMConv3d(gc, gc, (1, 5, 5), stride, dilation=(1, 3, 3), bias=bias, indice_key=f"{indice_key}_xy3")
        self.conv_channel = spconv.SubMConv3d(in_planes, in_planes, (1, 1, 1), stride, bias=bias, indice_key=f"{indice_key}_channel")
        # self.conv_channel = SparseConvMLP(in_planes, in_planes)
        self.conv_z = spconv.SubMConv3d(in_planes, gc, (7, 1, 1), stride, bias=bias, indice_key=f"{indice_key}_z")
        # self.mlp = SparseConvMLP(in_planes, in_planes, norm_layer=norm_fn)

    def forward(self, x):
        # shortcut = x
        x_z = self.conv_z(x)
        # feats = torch.cat([self.conv_xy1(x_z).features, self.conv_xy2(x_z).features, self.conv_xy3(x_z).features], 1)
        feats = torch.cat([x_z.features, self.conv_xy1(x_z).features, self.conv_xy2(x_z).features, self.conv_xy3(x_z).features], 1)
        # feats = x_z.features+self.conv_xy1(x_z).features+self.conv_xy2(x_z).features+self.conv_xy3(x_z).features
        x_xy = replace_feature(x, feats)
        x = self.conv_channel(x_xy)
        # x = replace_feature(x, feats+x.features)
        return x


class SparseConvMLP(spconv.SparseModule):
    def __init__(self, inplanes, planes, mlp_ratio=2, act_layer=nn.ReLU, norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), bias=None, indice_key=None):
        super().__init__()
        planes = planes or inplanes
        hidden_planes = int(inplanes*mlp_ratio)
        self.fc1 = spconv.SubMConv3d(inplanes, hidden_planes, 1, bias=bias, indice_key=f"{indice_key}_mlp")
        self.norm = norm_layer(hidden_planes)
        self.act = act_layer()

        self.fc2 = spconv.SubMConv3d(hidden_planes, planes, 1, bias=bias, indice_key=f"{indice_key}_mlp")

    def forward(self, x):
        x = self.fc1(x)
        x = replace_feature(x, self.norm(x.features))
        x = replace_feature(x, self.act(x.features))
        x = self.fc2(x)
        x = replace_feature(x, torch.mean(x.features, 1).unsqueeze(1))
        return x


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SWConv(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, bias=None, indice_key=None):
        super().__init__()
        k = kernel_size
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=(1, k, k), stride=stride, bias=bias, indice_key=f"{indice_key}_xy"
        )
        self.conv2 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=(k, 1, 1), stride=stride, bias=bias, indice_key=f"{indice_key}_z"
        )

    def forward(self, x):
        x = self.conv2(x)
        x = self.conv1(x)
        return x


class SparseBasicBlockV2(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlockV2, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        # self.conv1 = SWConv(
        #     inplanes, planes, kernel_size=7, stride=stride, bias=bias, indice_key=indice_key
        # )
        # self.conv1 = InceptionAtt(
        #     inplanes, kernel_size=3, stride=stride, bias=bias, norm_fn=norm_fn, indice_key=indice_key
        # )
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        # self.conv2 = SWConv(
        #     planes, planes, kernel_size=7, stride=stride, bias=bias, indice_key=indice_key
        # )
        # self.conv2 = InceptionAtt(
        #     planes, kernel_size=3, stride=stride, bias=bias, norm_fn=norm_fn, indice_key=indice_key
        # )
        self.conv2 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class BranchBlock(spconv.SparseModule):
    def __init__(self, block, in_channels, out_channels, use_bias, norm_fn, stride, padding, indice_key):
        super(BranchBlock, self).__init__()
        self.branch1 = spconv.SparseSequential(
                       spconv.SparseConv3d(in_channels, out_channels, 3, stride=stride, padding=padding[0],
                                            bias=False, indice_key=indice_key, dilation=1),
                       norm_fn(out_channels),
                       nn.ReLU(),
                       )
        self.branch2 = spconv.SparseSequential(
                       spconv.SparseConv3d(in_channels, out_channels, 3, stride=stride, padding=padding[1],
                                            bias=False, indice_key=indice_key, dilation=(1, 2, 2)),
                       norm_fn(out_channels),
                       nn.ReLU(),
                       )
        self.branch3 = spconv.SparseSequential(
                       spconv.SparseConv3d(in_channels, out_channels, 3, stride=stride, padding=padding[2],
                                            bias=False, indice_key=indice_key, dilation=(1, 3, 3)),
                       norm_fn(out_channels),
                       nn.ReLU(),
                       )
        self.conv = spconv.SparseSequential(
                SparseBasicBlock(out_channels, out_channels, bias=use_bias, norm_fn=norm_fn, indice_key=indice_key),
                SparseBasicBlock(out_channels, out_channels, bias=use_bias, norm_fn=norm_fn, indice_key=indice_key),
        )

    def forward(self, x):
        x_1 = self.branch1(x)
        x_2 = self.branch1(x)
        x_3 = self.branch1(x)
        x_1 = x_1.replace_feature(torch.cat([x_1.features, x_2.features, x_3.features]))
        x_1.indices = torch.cat([x_1.indices, x_2.indices, x_3.indices])
        x_1 = self.sparse_add(x_1)
        out = self.conv(x_1)
        return out
    
    def sparse_add(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices
        spatial_shape = x_conv.spatial_shape

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

class SparseBasicBlockSW(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlockSW, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = SWConv(
            inplanes, planes, kernel_size=3, stride=stride, bias=bias, indice_key=indice_key
        )
        # self.conv1 = InceptionAtt(
        #     inplanes, kernel_size=3, stride=stride, bias=bias, indice_key=indice_key
        # )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = SWConv(
            planes, planes, kernel_size=3, stride=stride, bias=bias, indice_key=indice_key
        )
        # self.conv2 = InceptionAtt(
        #     planes, kernel_size=3, stride=stride, bias=bias, indice_key=indice_key
        # )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, indice_key='subm1'),
            # InceptionAtt(16, kernel_size=3, norm_fn=norm_fn, indice_key='subm1')
            # SWConv(
            #     16, 16, kernel_size=3, stride=1, indice_key='subm1'
            # )
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            # InceptionAtt(32, kernel_size=3, norm_fn=norm_fn, indice_key='subm2'),
            # InceptionAtt(32, kernel_size=3, norm_fn=norm_fn, indice_key='subm2')
            block(32, 32, 3, norm_fn=norm_fn, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn,indice_key='subm2'),
            # SWConv(32, 32, kernel_size=3, stride=1, indice_key='subm2'),
            # SWConv(32, 32, kernel_size=3, stride=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, indice_key='subm3'),
            # InceptionAtt(64, kernel_size=3, norm_fn=norm_fn, indice_key='subm3'),
            # InceptionAtt(64, kernel_size=3, norm_fn=norm_fn, indice_key='subm3')
            # SWConv(64, 64, kernel_size=3, stride=1, indice_key='subm3'),
            # SWConv(64, 64, kernel_size=3, stride=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            # InceptionAtt(64, kernel_size=3, norm_fn=norm_fn, indice_key='subm4'),
            # InceptionAtt(64, kernel_size=3, norm_fn=norm_fn, indice_key='subm4')
            block(64, 64, 3, norm_fn=norm_fn, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, indice_key='subm4'),
            # SWConv(64, 64, kernel_size=3, stride=1, indice_key='subm4'),
            # SWConv(64, 64, kernel_size=3, stride=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        # self.conv3 = BranchBlock(block=block, in_channels=32, out_channels=64, use_bias=use_bias, norm_fn=norm_fn, stride=2,
        #                           padding=[1, 1, 1], indice_key='res3')

        # self.conv4 = BranchBlock(block=block, in_channels=64, out_channels=64, use_bias=use_bias, norm_fn=norm_fn, stride=2,
        #                           padding=[(0, 1, 1), (0, 2, 2), (0, 6, 6)], indice_key='res4')
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv4_add = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, dilation=(1, 2, 2), norm_fn=norm_fn, stride=2, padding=(0, 2, 2), indice_key='spconv4_add', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4_add'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4_add'),
        )

        self.conv4_add2 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, (3, 5, 5), dilation=(1, 3, 3), norm_fn=norm_fn, stride=2, padding=(0, 6, 6), indice_key='spconv4_add2', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4_add2'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res4_add2'),
        )

        # self.score_conv = spconv.SubMConv3d(
        #     128, 1, kernel_size=1
        # )

        self.share_conv = spconv.SparseSequential(
            spconv.SparseConv2d(64, 128, 3, padding=1, bias=False, indice_key='share_conv1'),
            norm_fn(128),
            nn.ReLU(),
            spconv.SparseConv2d(128, 128, 3, padding=1, bias=False, indice_key='share_conv2'),
            norm_fn(128),
            nn.ReLU(),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)

        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        # self.build_losses()

    def sparse_add(self, x_conv, downsample=False):
        features_cat = x_conv.features
        indices_cat = x_conv.indices
        spatial_shape = x_conv.spatial_shape
        if downsample:
            spatial_shape = [i // 2 for i in spatial_shape]

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

    def HeightComress(self, sparsetensor):
        features_cat = sparsetensor.features
        indices_cat = sparsetensor.indices[:, [0, 2, 3]]
        spatial_shape = sparsetensor.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=sparsetensor.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4_add = self.conv4_add(x_conv3)
        x_conv4_add2 = self.conv4_add2(x_conv3)
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv4_add.features, x_conv4_add2.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv4_add.indices, x_conv4_add2.indices])
        x_conv4 = self.sparse_add(x_conv4)

        # score_map = self.score_conv(x_conv4)

        # spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x_conv4)

        # if self.training:
        #     target_dict = self.assign_targets(
        #         batch_dict['gt_boxes'], num_voxels, spatial_indices, spatial_shape, x_conv4.features.shape[0]
        #     )
        #     self.forward_ret_dict['target_dicts'] = target_dict
        #     self.forward_ret_dict.update({'score_map': score_map.features})
        # score = self.sigmoid(score_map.features)
        # indices = torch.nonzero(score.squeeze() > self.thred).squeeze()
        # # print(len(indices))
        # if len(indices) <= 1:
        #     print('wrong')
        # x_conv4 = x_conv4.replace_feature(x_conv4.features[indices])
        # x_conv4.indices = x_conv4.indices[indices]
        
        # x_conv4 = x_conv4.replace_feature(x_conv4.features*self.sigmoid(score_map.features))

        out = self.HeightComress(x_conv4)

        out = self.share_conv(out)

        

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict

    # def _get_voxel_infos(self, x):
    #     spatial_shape = x.spatial_shape
    #     voxel_indices = x.indices
    #     spatial_indices = []
    #     num_voxels = []
    #     batch_size = x.batch_size
    #     batch_index = voxel_indices[:, 0]

    #     for bs_idx in range(batch_size):
    #         batch_inds = batch_index==bs_idx
    #         spatial_indices.append(voxel_indices[batch_inds][:, [3, 2, 1]])
    #         num_voxels.append(batch_inds.sum())

    #     return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels
    
    # def assign_targets(self, gt_boxes, num_voxels, spatial_indices, spatial_shape, nums):
    #     """
    #     Args:
    #         gt_boxes: (B, M, 8)
    #     Returns:
    #     """
    #     target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
    #     batch_size = gt_boxes.shape[0]
    #     heatmaps = torch.zeros([nums, 1]).to(gt_boxes.device)
    #     all_names = np.array(['bg', *self.class_names])
    #     ret_dict = {}

    #     for idx, cur_class_names in enumerate(self.class_names):
    #         heatmap_list = []
    #         for bs_idx in range(batch_size):
    #             cur_gt_boxes = gt_boxes[bs_idx]
    #             gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

    #             gt_boxes_single_head = []

    #             for idx, name in enumerate(gt_class_names):
    #                 if name not in cur_class_names:
    #                     continue
    #                 temp_box = cur_gt_boxes[idx]
    #                 temp_box[-1] = cur_class_names.index(name) + 1
    #                 gt_boxes_single_head.append(temp_box[None, :])

    #             if len(gt_boxes_single_head) == 0:
    #                 gt_boxes_single_head = cur_gt_boxes[:0, :]
    #             else:
    #                 gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

    #             heatmap = self.assign_target_of_single_head(
    #                 gt_boxes=gt_boxes_single_head,
    #                 num_voxels=num_voxels[bs_idx], spatial_indices=spatial_indices[bs_idx], 
    #                 spatial_shape=spatial_shape, 
    #                 feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
    #                 num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
    #                 gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
    #                 min_radius=target_assigner_cfg.MIN_RADIUS,
    #             )
    #             heatmap_list.append(heatmap.to(gt_boxes_single_head.device))

    #         heatmap_cur_class = torch.cat(heatmap_list, dim=1).permute(1, 0)
    #         heatmaps += heatmap_cur_class

    #     ret_dict['heatmaps'] = heatmaps

    #     return ret_dict
    
    # def assign_target_of_single_head(
    #         self, gt_boxes, num_voxels, spatial_indices, spatial_shape, feature_map_stride, num_max_objs=500,
    #         gaussian_overlap=0.1, min_radius=2
    # ):
    #     """
    #     Args:
    #         gt_boxes: (N, 8)
    #         feature_map_size: (2), [x, y]

    #     Returns:

    #     """
    #     heatmap = gt_boxes.new_zeros(1, num_voxels)

    #     # ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
    #     inds = gt_boxes.new_zeros(num_max_objs).long()
    #     # mask = gt_boxes.new_zeros(num_max_objs).long()

    #     x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
    #     coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
    #     coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
    #     coord_z = (z - self.point_cloud_range[2]) / self.voxel_size[2] / feature_map_stride

    #     coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[2] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
    #     coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[1] - 0.5)  #
    #     coord_z = torch.clamp(coord_z, min=0, max=spatial_shape[0] - 0.5)

    #     center = torch.cat((coord_x[:, None], coord_y[:, None], coord_z[:, None]), dim=-1)
    #     center_int = center.int()
    #     # center_int_float = center_int.float()

    #     dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
    #     dx = dx / self.voxel_size[0] / feature_map_stride
    #     dy = dy / self.voxel_size[1] / feature_map_stride
    #     dz = dz / self.voxel_size[2] / feature_map_stride

    #     radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
    #     radius = torch.clamp_min(radius.int(), min=min_radius)

    #     for k in range(min(num_max_objs, gt_boxes.shape[0])):
    #         if dx[k] <= 0 or dy[k] <= 0:
    #             continue

    #         if not (0 <= center_int[k][0] <= spatial_shape[2] and 0 <= center_int[k][1] <= spatial_shape[1] 
    #                 and 0 <= center_int[k][2] <= spatial_shape[0]):
    #             continue

    #         # cur_class_id = (gt_boxes[k, -1] - 1).long()
    #         distance = self.distance(spatial_indices, center[k])
    #         inds[k] = distance.argmin()
    #         # mask[k] = 1

    #         if 'gt_center' in self.gaussian_type:
    #             centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap, distance, radius[k].item() * self.gaussian_ratio)

    #         if 'nearst' in self.gaussian_type:
    #             centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap, self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * self.gaussian_ratio)

    #     return heatmap

    # def distance(self, voxel_indices, center):
    #     distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
    #     return distances

    # def sigmoid(self, x):
    #     y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    #     return y
    
    # def build_losses(self):
    #     self.add_module('score_loss_func', loss_utils.FocalLoss())

    # def get_loss(self):
    #     loss = 0
    #     score_loss = self.score_loss_func(self.sigmoid(self.forward_ret_dict['score_map']), self.forward_ret_dict['target_dicts']['heatmaps'])
    #     score_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['score_weight']
    #     tb_dict = {'score_loss': score_loss.item()}
    #     loss += score_loss
    #     return loss, tb_dict


class VoxelResBackBone8xOri(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict

