import torch
import random


def cluster_sample(bs_first, features, coords, cluster_mask, k, spatial_shape):
    cluster_num = cluster_mask.shape[0]
    features_list = []
    coords_list = []
    index_list = []
    relative_coords_list = []
    key_mask_list = []
    for clt in range(cluster_num):
        mask = cluster_mask[clt] == 1
        per_cls_features = features[mask]
        per_cls_coords = coords[mask]
        # per_cls_features = torch.cat([features[mask], features[bs_first][clt, ...].unsqueeze(0)])
        # per_cls_coords = torch.cat([coords[mask], coords[bs_first][clt].unsqueeze(0)])
        per_cls_index = torch.nonzero(cluster_mask[clt])


        per_cls_features, per_cls_coords, per_cls_index, per_cle_relative_coords, per_cls_mask = random_sample(per_cls_features, per_cls_coords, per_cls_index, k, spatial_shape)
        
        features_list.append(per_cls_features)
        coords_list.append(per_cls_coords)
        index_list.append(per_cls_index)
        relative_coords_list.append(per_cle_relative_coords)
        key_mask_list.append(per_cls_mask)
    
    cls_features = torch.stack(features_list)
    cls_coords = torch.stack(coords_list)
    cls_index = torch.cat(index_list, 0)
    cls_relative_coords = torch.stack(relative_coords_list)
    cls_key_mask = torch.stack(key_mask_list)

    return cls_features, cls_coords, cls_index, cls_relative_coords, cls_key_mask


def random_sample(features, coords, index, k, spatial_shape):
    """
    return:
    cluster features
    cluster coords
    cluster_relative coords
    mask: 1表示selected  -1表示padding
    """
    num_ele = coords.shape[0]
    if num_ele <= k:
        relative_coords = coords.clone().to(torch.float32)
        centroid_coord = relative_coords[:, 1:].mean(0).to(torch.int32)
        relative_coords[:, 1:] = coords[:, 1:] - centroid_coord
        relative_coords[:, 1:] /= spatial_shape
        padding = features[-1,:].unsqueeze(0).repeat(k-num_ele, 1)
        features = torch.cat([features, padding], 0)
        padding = coords[-1,:].unsqueeze(0).repeat(k-num_ele, 1)
        coords = torch.cat([coords, padding], 0)
        padding = relative_coords[-1,:].unsqueeze(0).repeat(k-num_ele, 1)
        relative_coords = torch.cat([relative_coords, padding], 0)
        # padding = index[-1,:].unsqueeze(0).repeat(k-num_ele, 1)
        # index = torch.cat([index, padding], 0)
        mask = torch.Tensor([1 for i in range(num_ele)] + [-1 for i in range(k-num_ele)]).to(coords.device)
        assert (mask==1).sum() == index.shape[0]

    else:
        idx = random.sample(range(1, coords.shape[0]), k)
        features = features[idx]
        coords = coords[idx]
        index = index[idx]
        relative_coords = coords.clone().to(torch.float32)
        centroid_coord = relative_coords[:, 1:].mean(0).to(torch.int32)
        relative_coords[:, 1:] = coords[:, 1:] - centroid_coord
        relative_coords[:, 1:] /= spatial_shape
        mask = torch.Tensor([1 for i in range(coords.shape[0])]).to(coords.device)
    

    return features, coords, index, relative_coords, mask