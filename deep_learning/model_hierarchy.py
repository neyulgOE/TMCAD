import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    x: (B, N, C)
    returns: (B, N, k) indices
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=-1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    x: (B, C, N)
    returns: (B, 2*C, N, k)
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)

    x_trans = x.transpose(2, 1).contiguous()  # (B, N, C)
    if idx is None:
        idx = knn(x_trans, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x_trans.view(batch_size * num_points, -1)
    feature = x_flat[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_expanded = x_trans.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    diff = feature - x_expanded

    if num_dims == 3:
        dist = torch.norm(diff, dim=-1, keepdim=True)
        feature = torch.cat((x_expanded, diff, dist), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((x_expanded, diff), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class CADPatchEncoder(nn.Module):
    """
    Encodes local topological K-ring patches from Canonical CAD Hierarchy into descriptors.
    Input:
      - patch_features: (B, N_cad, K, 13)
      - patch_rel_pos:  (B, N_cad, K, 3)
      - patch_hops:     (B, N_cad, K)
    Output:
      - D_cad: (B, N_cad, descriptor_dim) L2-normalized
    """
    def __init__(self, descriptor_dim: int = 128, patch_k: int = 32):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.patch_k = patch_k

        in_dim = 13 + 3 + 1  # 17 dimensions (features13 + rel_xyz + hop)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, descriptor_dim)
        )

    def forward(self, patch_features, patch_rel_pos, patch_hops):
        # Flatten batch and key primitive dims: (B * N_cad, K, C)
        B, N_cad, K, _ = patch_features.shape
        hops_norm = (patch_hops.unsqueeze(-1).float()) / float(self.patch_k)

        # x: (B, N_cad, K, 17)
        min_m = min(patch_features.shape[1], patch_rel_pos.shape[1], hops_norm.shape[1])
        if patch_features.shape[1] != min_m or patch_rel_pos.shape[1] != min_m or hops_norm.shape[1] != min_m:
            patch_features = patch_features[:, :min_m, :, :]
            patch_rel_pos = patch_rel_pos[:, :min_m, :, :]
            hops_norm = hops_norm[:, :min_m, :, :]
            N_cad = min_m

        x = torch.cat([patch_features, patch_rel_pos, hops_norm], dim=-1)
        x = x.view(B * N_cad * K, -1)

        feat = self.mlp1(x) # (B * N_cad * K, 256)
        feat = feat.view(B * N_cad, K, 256)

        max_pool = torch.max(feat, dim=1)[0]   # (B * N_cad, 256)
        mean_pool = torch.mean(feat, dim=1)    # (B * N_cad, 256)
        fused = torch.cat([max_pool, mean_pool], dim=-1) # (B * N_cad, 512)

        desc = self.mlp2(fused) # (B * N_cad, descriptor_dim)
        desc = desc.view(B, N_cad, self.descriptor_dim)
        return F.normalize(desc, p=2, dim=-1)

class ObservationEncoder(nn.Module):
    """
    DGCNN-based Point Cloud Encoder for partial observation (depth image -> camera points).
    Input:
      - obs_cam_pts: (B, N_obs, 3)
    Output:
      - D_obs: (B, N_obs, descriptor_dim) L2-normalized
    """
    def __init__(self, k: int = 20, descriptor_dim: int = 128):
        super().__init__()
        self.k = k
        self.descriptor_dim = descriptor_dim

        # EdgeConv 1: 7 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # EdgeConv 2: 64*2 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # EdgeConv 3: 128*2 -> 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64 + 128 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.mlp_out = nn.Sequential(
            nn.Conv1d(512 + 512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, descriptor_dim, kernel_size=1)
        )

    def forward(self, obs_pts):
        # obs_pts: (B, N, 3) -> (B, 3, N)
        x = obs_pts.transpose(2, 1).contiguous()
        B, C, N = x.shape

        x1 = get_graph_feature(x, k=self.k) # (B, 7, N, k)
        x1 = self.conv1(x1)                 # (B, 64, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        x2 = get_graph_feature(x1, k=self.k) # (B, 128, N, k)
        x2 = self.conv2(x2)                  # (B, 128, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0] # (B, 128, N)

        x3 = get_graph_feature(x2, k=self.k) # (B, 256, N, k)
        x3 = self.conv3(x3)                  # (B, 256, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0] # (B, 256, N)

        cat_feat = torch.cat((x1, x2, x3), dim=1) # (B, 448, N)
        point_feat = self.conv4(cat_feat)         # (B, 512, N)

        global_feat = point_feat.max(dim=-1, keepdim=True)[0].repeat(1, 1, N) # (B, 512, N)
        fused = torch.cat((point_feat, global_feat), dim=1) # (B, 1024, N)

        desc = self.mlp_out(fused) # (B, descriptor_dim, N)
        desc = desc.transpose(2, 1).contiguous() # (B, N, descriptor_dim)
        return F.normalize(desc, p=2, dim=-1)

class CADHierarchyCorrespondenceModel(nn.Module):
    """
    Full CAD-aware Hierarchical Correspondence Learning Network:
      CADPatchEncoder + ObservationEncoder + Cosine Similarity Matrix.
    """
    def __init__(self, descriptor_dim: int = 128, patch_k: int = 32, obs_k: int = 20, temperature: float = 0.07):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.temperature = temperature

        self.cad_encoder = CADPatchEncoder(descriptor_dim=descriptor_dim, patch_k=patch_k)
        self.obs_encoder = ObservationEncoder(k=obs_k, descriptor_dim=descriptor_dim)

    def encode_cad(self, patch_features, patch_rel_pos, patch_hops):
        return self.cad_encoder(patch_features, patch_rel_pos, patch_hops)

    def encode_obs(self, obs_pts):
        return self.obs_encoder(obs_pts)

    def compute_similarity(self, D_obs, D_cad):
        """
        D_obs: (B, N_obs, D)
        D_cad: (B, N_cad, D)
        Returns S: (B, N_obs, N_cad) scaled by temperature
        """
        sim = torch.bmm(D_obs, D_cad.transpose(2, 1)) # (B, N_obs, N_cad)
        return sim / self.temperature

    def forward(self, obs_pts, patch_features, patch_rel_pos, patch_hops):
        D_obs = self.encode_obs(obs_pts)
        D_cad = self.encode_cad(patch_features, patch_rel_pos, patch_hops)
        S = self.compute_similarity(D_obs, D_cad)
        return S, D_obs, D_cad
