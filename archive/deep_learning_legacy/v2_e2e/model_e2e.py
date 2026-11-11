import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_learning.procrustes import differentiable_kabsch


def knn_points(p, k):
    """
    p: (B, N, 3)
    returns: (B, N, k) indices
    """
    inner = -2 * torch.matmul(p, p.transpose(2, 1))
    pp = torch.sum(p ** 2, dim=-1, keepdim=True)
    pairwise_distance = -pp - inner - pp.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_6d_graph_feature(x, k=20, idx=None):
    """
    Constructs edge features using both 3D coordinates and 3D surface normals.
    x: (B, 6, N) where x[:, :3, :] is position p, x[:, 3:6, :] is normal n.
    returns: (B, 15, N, k)
    """
    B, C, N = x.shape
    p = x[:, :3, :].transpose(2, 1).contiguous()  # (B, N, 3)
    n = x[:, 3:6, :].transpose(2, 1).contiguous() # (B, N, 3)

    if idx is None:
        idx = knn_points(p, k=k)  # (B, N, k) based on Euclidean distance of position

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)

    # Position features
    p_flat = p.view(B * N, 3)
    p_neighbors = p_flat[idx_flat, :].view(B, N, k, 3)
    p_center = p.view(B, N, 1, 3).repeat(1, 1, k, 1)
    p_diff = p_neighbors - p_center
    dist = torch.norm(p_diff, dim=-1, keepdim=True)
    unit_p_diff = p_diff / (dist + 1e-6)

    # Normal features
    n_flat = n.view(B * N, 3)
    n_neighbors = n_flat[idx_flat, :].view(B, N, k, 3)
    n_center = n.view(B, N, 1, 3).repeat(1, 1, k, 1)
    n_diff = n_neighbors - n_center
    n_dot = torch.sum(n_center * n_neighbors, dim=-1, keepdim=True)
    n_proj = torch.sum(n_center * unit_p_diff, dim=-1, keepdim=True)

    # Cat: [p_center(3), p_diff(3), dist(1), n_center(3), n_diff(3), n_dot(1), n_proj(1)] = 15
    edge_feat = torch.cat(
        (p_center, p_diff, dist, n_center, n_diff, n_dot, n_proj),
        dim=3
    ).permute(0, 3, 1, 2).contiguous()  # (B, 15, N, k)

    return edge_feat, idx


def get_feature_graph(x, k=20, idx=None):
    """
    Standard EdgeConv feature graph on intermediate feature maps.
    x: (B, C, N)
    returns: (B, 2*C, N, k)
    """
    B, C, N = x.shape
    x_trans = x.transpose(2, 1).contiguous()
    if idx is None:
        idx = knn_points(x_trans, k=k)

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)

    x_flat = x_trans.view(B * N, C)
    feature = x_flat[idx_flat, :].view(B, N, k, C)
    x_expanded = x_trans.view(B, N, 1, C).repeat(1, 1, k, 1)
    diff = feature - x_expanded

    edge_feat = torch.cat((x_expanded, diff), dim=3).permute(0, 3, 1, 2).contiguous()
    return edge_feat


class GeometricDGCNN_6D(nn.Module):
    """
    DGCNN backbone that ingests 6D point inputs (x, y, z, nx, ny, nz)
    and extracts multi-scale local + global point features.
    Output: (B, feature_dim, N)
    """
    def __init__(self, k=20, out_dim=128):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

        # EdgeConv 1: 15 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # EdgeConv 2: 64*2 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # EdgeConv 3: 64*2 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Aggregation: 64 + 64 + 128 = 256
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fusion: local (256) + global (256) = 512 -> out_dim
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """
        x: (B, 6, N)
        returns: (B, out_dim, N)
        """
        N = x.size(2)

        # 1st EdgeConv on 6D coordinates + normals
        e1, idx1 = get_6d_graph_feature(x, k=self.k)  # (B, 15, N, k)
        e1 = self.conv1(e1)                           # (B, 64, N, k)
        e1 = e1.max(dim=-1, keepdim=False)[0]         # (B, 64, N)

        # 2nd EdgeConv
        e2 = get_feature_graph(e1, k=self.k)          # (B, 128, N, k)
        e2 = self.conv2(e2)                           # (B, 64, N, k)
        e2 = e2.max(dim=-1, keepdim=False)[0]         # (B, 64, N)

        # 3rd EdgeConv
        e3 = get_feature_graph(e2, k=self.k)          # (B, 128, N, k)
        e3 = self.conv3(e3)                           # (B, 128, N, k)
        e3 = e3.max(dim=-1, keepdim=False)[0]         # (B, 128, N)

        # Concatenate multi-scale local features
        local_feat = torch.cat((e1, e2, e3), dim=1)   # (B, 256, N)
        local_feat = self.conv4(local_feat)           # (B, 256, N)

        # Global context
        global_feat = local_feat.max(dim=-1, keepdim=True)[0]  # (B, 256, 1)
        global_feat = global_feat.repeat(1, 1, N)              # (B, 256, N)

        # Combined point features
        combined = torch.cat((local_feat, global_feat), dim=1) # (B, 512, N)
        out = self.conv5(combined)                             # (B, out_dim, N)

        return out


class KeypointVotingModule(nn.Module):
    """
    Predicts 3D offset vectors and confidence weights from observed points
    to K predefined object keypoints, then computes differentiable weighted voting.
    """
    def __init__(self, in_dim=128, num_keypoints=8):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Offset head: predicts (dx, dy, dz) for each of the K keypoints
        self.offset_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, num_keypoints * 3, kernel_size=1)
        )

        # Confidence weight head: predicts weight in [0, 1] for each keypoint
        self.weight_conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, num_keypoints, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat, pts_xyz):
        """
        feat: (B, in_dim, N) Point features
        pts_xyz: (B, 3, N) Point 3D positions in camera frame
        Returns:
            voted_keypoints: (B, K, 3) Camera frame keypoints
            keypoint_weights: (B, K) Overall confidence for each keypoint
            offsets: (B, K, 3, N) Raw offset predictions
            pt_weights: (B, K, N) Per-point confidence weights
        """
        B, _, N = pts_xyz.shape
        K = self.num_keypoints

        offsets = self.offset_conv(feat)  # (B, K*3, N)
        offsets = offsets.view(B, K, 3, N)

        weights = self.weight_conv(feat)  # (B, K, N)

        # For each point i and keypoint k: candidate = pts_xyz[:, :, i] + offsets[:, k, :, i]
        pts_expanded = pts_xyz.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, 3, N)
        candidates = pts_expanded + offsets                     # (B, K, 3, N)

        # Weighted voting: (B, K, 3, N) * (B, K, 1, N) -> sum over N
        w_expanded = weights.unsqueeze(2)                       # (B, K, 1, N)
        w_sum = torch.sum(w_expanded, dim=-1, keepdim=True) + 1e-6  # (B, K, 1, 1)

        voted_kp = torch.sum(candidates * w_expanded, dim=-1) / w_sum.squeeze(-1)  # (B, K, 3)

        # Overall keypoint confidence: mean weight of top confident voting points
        kp_weights = torch.mean(weights, dim=-1)  # (B, K)

        return voted_kp, kp_weights, offsets, weights


class GeoPoseNet(nn.Module):
    """
    End-to-End 6DoF Geometric Pose Estimation Network.
    Integrates:
      1. 6D Normal-Aware DGCNN Backbone
      2. Keypoint Offset Voting Module (Coarse-to-Fine)
      3. Differentiable Kabsch SVD (Procrustes) Layer
    """
    def __init__(self, cad_keypoints, k_neighbors=20, feat_dim=128):
        """
        cad_keypoints: (K, 3) tensor or numpy array of predefined CAD keypoints
        """
        super().__init__()
        if not isinstance(cad_keypoints, torch.Tensor):
            cad_keypoints = torch.tensor(cad_keypoints, dtype=torch.float32)
        self.register_buffer("cad_keypoints", cad_keypoints)  # (K, 3)

        self.num_keypoints = cad_keypoints.shape[0]
        self.backbone = GeometricDGCNN_6D(k=k_neighbors, out_dim=feat_dim)
        self.voting_head = KeypointVotingModule(in_dim=feat_dim, num_keypoints=self.num_keypoints)

    def forward(self, obs_points_6d):
        """
        obs_points_6d: (B, N, 6) or (B, 6, N) where channels are (x, y, z, nx, ny, nz)
        Returns dictionary containing:
            R: (B, 3, 3) Predicted rotation matrix
            t: (B, 3) Predicted translation vector
            voted_kp: (B, K, 3) Voted camera-space keypoints
            kp_weights: (B, K) Confidence weights
        """
        if obs_points_6d.size(1) != 6 and obs_points_6d.size(2) == 6:
            obs_points_6d = obs_points_6d.transpose(1, 2).contiguous()  # (B, 6, N)

        B, _, N = obs_points_6d.shape
        pts_xyz = obs_points_6d[:, :3, :]  # (B, 3, N)

        # 1. Extract 6D geometric features
        feat = self.backbone(obs_points_6d)  # (B, feat_dim, N)

        # 2. Keypoint voting
        voted_kp, kp_weights, offsets, pt_weights = self.voting_head(feat, pts_xyz)

        # 3. Differentiable Kabsch SVD: Solve R, t from CAD keypoints -> Voted camera keypoints
        # Repeat CAD keypoints across batch
        cad_kp_batch = self.cad_keypoints.unsqueeze(0).repeat(B, 1, 1)  # (B, K, 3)
        R, t = differentiable_kabsch(cad_kp_batch, voted_kp, weights=kp_weights)

        return {
            "R": R,
            "t": t,
            "voted_keypoints": voted_kp,
            "keypoint_weights": kp_weights,
            "offsets": offsets,
            "pt_weights": pt_weights
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing GeoPoseNet on:", device)

    # 8 synthetic CAD keypoints (corners of a box)
    cad_kp = torch.tensor([
        [-0.2, -0.4, -0.2], [ 0.2, -0.4, -0.2],
        [-0.2,  0.4, -0.2], [ 0.2,  0.4, -0.2],
        [-0.2, -0.4,  0.2], [ 0.2, -0.4,  0.2],
        [-0.2,  0.4,  0.2], [ 0.2,  0.4,  0.2]
    ], dtype=torch.float32)

    model = GeoPoseNet(cad_keypoints=cad_kp, k_neighbors=16, feat_dim=128).to(device)

    # Dummy input: (B=2, N=1024, 6)
    dummy_input = torch.randn(2, 1024, 6, device=device)
    out = model(dummy_input)

    print("R shape:", out["R"].shape)
    print("t shape:", out["t"].shape)
    print("voted_keypoints shape:", out["voted_keypoints"].shape)
    print("keypoint_weights shape:", out["keypoint_weights"].shape)

    assert out["R"].shape == (2, 3, 3)
    assert out["t"].shape == (2, 3)
    assert out["voted_keypoints"].shape == (2, 8, 3)

    # Test backward pass
    loss = out["R"].sum() + out["t"].sum() + out["voted_keypoints"].sum()
    loss.backward()
    print("Backward pass PASSED! GeoPoseNet is fully differentiable!")
