import torch
import torch.nn as nn
import torch.nn.functional as F


class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super(VotingModule, self).__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = nn.Conv1d(self.in_dim, (2 + self.out_dim) * self.vote_factor, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(self.in_dim)
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU()
        )

    def forward(self, seed_xy, seed_features):
        """ Forward pass.

        Arguments:
            seed_xy: (batch_size, num_seed, 2) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xy: (batch_size, num_seed*vote_factor, 2)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xy.shape[0]
        num_seed = seed_xy.shape[1]
        num_vote = num_seed * self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 2 + self.out_dim)
        offset = net[:, :, :, 0:2]
        vote_xy = seed_xy.unsqueeze(2)
        vote_xy[:, :, :, 1:] = vote_xy[:, :, :, 1:] + offset
        vote_xy = vote_xy.contiguous().view(batch_size, num_vote, 3)

        residual_features = net[:, :, :, 2:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()
        return vote_xy, vote_features
