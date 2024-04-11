import torch
import torch.nn as nn
import numpy as np


from ...utils import box_coder_utils, common_utils, loss_utils, box_utils
from ...utils.spconv_utils import replace_feature, spconv
from ...ops.iou3d_nms import iou3d_nms_utils


class AnchorHeadSparseSingle(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_anchors_per_location = 2
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )
        self.match_height = anchor_target_cfg.MATCH_HEIGHT
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        self.anchor_generator_cfg = anchor_generator_cfg
        # anchors, self.num_anchors_per_location = self.generate_anchors(
        #     anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
        #     anchor_ndim=self.box_coder.code_size
        # )
        
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.class_names = np.array(class_names)

        self.num_anchors_per_location = [len(x['anchor_sizes'])*len(x['anchor_rotations'])*len(x['anchor_bottom_heights']) for x in anchor_generator_cfg]
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.conv_cls = spconv.SubMConv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = spconv.SubMConv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = spconv.SubMConv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        # self.init_weights()
        self.forward_ret_dict = {}
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)

        self.build_losses(self.model_cfg.LOSS_CONFIG)

    # def init_weights(self):
    #     pi = 0.01
    #     nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
    #     nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLossSparse()
        )

    def _get_voxel_infos(self, x):
        spatial_shape = [x.spatial_shape[1], x.spatial_shape[0]]
        voxel_indices = x.indices[:, [0, 2, 1]]
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels
    
    def generate_anchors_sparse(self, voxel_indices):
        grid_sizes = [self.grid_size[:2] // config['feature_map_stride'] for config in self.anchor_generator_cfg]
        anchor_sizes = [config['anchor_sizes'] for config in self.anchor_generator_cfg]
        anchor_rotations = [config['anchor_rotations'] for config in self.anchor_generator_cfg]
        anchor_heights = [config['anchor_bottom_heights'] for config in self.anchor_generator_cfg]
        align_center = [config.get('align_center', False) for config in self.anchor_generator_cfg]
        num_of_anchor_sets = len(anchor_sizes)
        assert len(grid_sizes) == num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, anchor_sizes, anchor_rotations, anchor_heights, align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.point_cloud_range[3] - self.point_cloud_range[0]) / grid_size[0]
                y_stride = (self.point_cloud_range[4] - self.point_cloud_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.point_cloud_range[3] - self.point_cloud_range[0]) / (grid_size[0] - 1)
                y_stride = (self.point_cloud_range[4] - self.point_cloud_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0
            """修改x_shitfs, y_shifts"""
            # x_shifts = torch.arange(
            #     self.point_cloud_range[0] + x_offset, self.point_cloud_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            # ).cuda()
            x_shifts = (voxel_indices[:, 1] * x_stride) + self.point_cloud_range[0]
            # y_shifts = torch.arange(
            #     self.point_cloud_range[1] + y_offset, self.point_cloud_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            # ).cuda()
            y_shifts = (voxel_indices[:, 2] * y_stride) + self.point_cloud_range[1]
            z_shifts = x_shifts.new_tensor(anchor_height).repeat(x_shifts.shape[0])

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            # x_shifts, y_shifts, z_shifts = torch.meshgrid([
            #     x_shifts, y_shifts, z_shifts
            # ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack([voxel_indices[:, 0], x_shifts, y_shifts, z_shifts]).T
            anchors = anchors[:, None, None, :].repeat(1, 1, num_anchor_size, 1) # [NV, C, Ns, 4]
            anchor_size = anchor_size.view(1, 1, -1, 3).repeat([*anchors.shape[0:2], 1, 1]) 
            anchors = torch.cat([anchors, anchor_size], -1) # [NV, C, Ns, 7]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, num_anchor_rotation, 1) # [NV, C, Ns, Nr, 7]
            anchor_rotation = anchor_rotation.view(1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat([anchors, anchor_rotation], -1)

            anchors = anchors.permute(1, 0, 2, 3, 4).contiguous() # [C, Nv, Ns, Nr, 7]
            anchors[..., 3] += anchors[..., 6] / 2  # shift to box centers
            all_anchors.append(anchors)
            # all_anchors = [x.cuda() for x in all_anchors]
        return all_anchors
    
    def forward(self, data_dict):
        # spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = data_dict['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(spatial_features_2d)
        self.forward_ret_dict['batch_index'] = batch_index

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds.features
        self.forward_ret_dict['box_preds'] = box_preds.features

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds.features
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                data_dict['gt_boxes'], spatial_indices, spatial_shape, num_voxels, voxel_indices
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_anchor_index = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'], voxel_indices=voxel_indices,
                cls_preds=cls_preds.features, box_preds=box_preds.features, dir_cls_preds=dir_cls_preds.features
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_index'] = batch_anchor_index
            data_dict['cls_preds_normalized'] = False

        return data_dict
    
    def assign_targets(self, gt_boxes_with_classes, spatial_indices, spatial_shape, num_voxels, voxel_indices):

        self.anchors = self.generate_anchors_sparse(voxel_indices)

        bbox_targets = []
        cls_labels = []
        reg_weights = []
        bs_anchor_num = []
        self.sparse_anchors = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            bs_spatial_indices = spatial_indices[k]
            bs_num_voxel = num_voxels[k]
            """remove zero anchor"""
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            """
            针对每类anchor进行target assign
            """
            for anchor_class_name, anchors in zip(self.anchor_class_names, self.anchors):
                """mask: acquire aimed class gt anchor"""
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)
                    
                anchors = anchors[anchors[..., 0]==k, :] # [num_anchor, 8]
                self.sparse_anchors.append(anchors)

                if self.use_multihead:
                    raise NotImplementedError
                    # anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    # selected_classes = cur_gt_classes[mask]
                else:
                    # feature_map_size = anchors.shape[:3]
                    # feature_map_size = spatial_shape
                    anchors = anchors.view(-1, anchors.shape[-1])  # (num_anchor, bs_index+7)
                    selected_classes = cur_gt_classes[mask]
                """对每一个类别生成 label, anchor_offset, weight"""
                single_target = self.assign_targets_single(
                    anchors[..., 1:],
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                raise NotImplementedError
                # target_dict = {
                #     'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                #     'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                #     'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                # }

                # target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                # target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                # target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(bs_num_voxel, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(bs_num_voxel, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(bs_num_voxel, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
            bs_anchor_num.append(target_dict['box_reg_targets'].shape[0])

        bbox_targets = torch.cat(bbox_targets, dim=0)

        cls_labels = torch.cat(cls_labels, dim=0)
        reg_weights = torch.cat(reg_weights, dim=0)
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
            'bs_anchor_nums': bs_anchor_num,

        }
        return all_targets_dict
        
    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            # NOTE: The speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            """
            anchor_to_gt_argmax: 每个anchor与gt最大的iou对应的gt索引
            anchor_to_gt_max: 每个anchor与gt最大的iou
            """
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            # gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            """anchors_with_max_overlap: gt对应的anchor索引
            gt_inds_force:上一步的anchor对应的gt索引
            """
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()
            """
            pos_inds: anchor是否为正例
            gt_inds_over_thresh: 正例anchor对应的索引
            bg_inds: 背景框
            """
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict

    def generate_predicted_boxes(self, batch_size, voxel_indices, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        self.anchors = self.generate_anchors_sparse(voxel_indices)
        if isinstance(self.anchors, list):
            if self.use_multihead:
                raise NotImplementedError
                # anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                #                      for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(-1, anchors.shape[-1])
        batch_index = batch_anchors[:, 0]
        batch_cls_preds = cls_preds.view(num_anchors, -1) \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors[:, 1:])

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds, batch_index

    def _pos_nomalizer(self, positives, bs_anchor_nums):
        split_pos = torch.split(positives, bs_anchor_nums)
        pos_nomalizer = []
        for idx, bs_pos in enumerate(split_pos):
            pos_nomalizer.append(bs_pos.sum(-1).float().unsqueeze(0).repeat(bs_anchor_nums[idx]))
        pos_nomalizer = torch.cat(pos_nomalizer, 0)
        # pos_nomalizer = pos_nomalizer.repeat(*bs_anchor_nums)
        # bs_anchor_nums.insert(0, 0)
        # pos_nomalizer = []
        # for i in range(len(bs_anchor_nums)-1):
        #     per_anchor_nums = positives[bs_anchor_nums[i]:bs_anchor_nums[i+1]].sum(0, keepdim=True).float()
        #     pos_nomalizer.append(per_anchor_nums.repeat(bs_anchor_nums[i+1]))

        return pos_nomalizer


    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds'] # b, h, w, c
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # b, 
        batch_size = len(self.forward_ret_dict['bs_anchor_nums'])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        """pos_normalizer需要修改, 生成的值可能有问题"""
        pos_normalizer = self._pos_nomalizer(positives, self.forward_ret_dict['bs_anchor_nums'])
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(-1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:] # b, n, 3      aim  N, 3
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict
    
    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = len(self.forward_ret_dict['bs_anchor_nums'])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = self._pos_nomalizer(positives, self.forward_ret_dict['bs_anchor_nums'])
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                raise NotImplementedError
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(-1, anchors.shape[-1])
        # anchors = self.sparse_anchors

        box_preds = box_preds.view(-1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors[..., 1:], box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )
            """从这开始改"""
            dir_logits = box_dir_cls_preds.view(-1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= self._pos_nomalizer(weights, self.forward_ret_dict['bs_anchor_nums'])
            weights /= torch.clamp(weights, min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict
    
    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2
    
    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        # batch_size = reg_targets.shape[0]
        # anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict
    