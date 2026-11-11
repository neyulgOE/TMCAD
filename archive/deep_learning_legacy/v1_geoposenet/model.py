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
        # Include SO(3) invariant features: pairwise distance, radial distance, and angle
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_c = torch.norm(x_expanded, dim=-1, keepdim=True)
        cos_ang = torch.sum(x_expanded * diff, dim=-1, keepdim=True) / (dist_c * dist + 1e-6)
        feature = torch.cat((x_expanded, diff, dist, dist_c, cos_ang), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((x_expanded, diff), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (B, 9, N, k) for first layer, (B, 2*C, N, k) for subsequent layers

class GeometricDGCNN(nn.Module):
    """
    Dynamic Graph CNN (DGCNN) for extracting local 3D geometric descriptors.
    Maps (B, N, 3) -> (B, N, descriptor_dim) with L2-normalization.
    """
    def __init__(self, k=20, descriptor_dim=32):
        super().__init__()
        self.k = k
        self.descriptor_dim = descriptor_dim

        # EdgeConv Block 1: 9 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # EdgeConv Block 2: 64 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # EdgeConv Block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Multi-scale aggregation: 64 + 64 + 128 = 256
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Fusion of local (256) + global (256) context = 512
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, descriptor_dim, kernel_size=1, bias=True)
        )

    def forward(self, x):
        """
        x: (B, N, 3) point coordinates
        returns: (B, N, descriptor_dim) L2-normalized unit descriptors
        """
        # (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1).contiguous()
        num_points = x.size(2)

        # EdgeConv 1
        e1 = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        e1 = self.conv1(e1)                  # (B, 64, N, k)
        e1 = e1.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # EdgeConv 2
        e2 = get_graph_feature(e1, k=self.k) # (B, 128, N, k)
        e2 = self.conv2(e2)                  # (B, 64, N, k)
        e2 = e2.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # EdgeConv 3
        e3 = get_graph_feature(e2, k=self.k) # (B, 128, N, k)
        e3 = self.conv3(e3)                  # (B, 128, N, k)
        e3 = e3.max(dim=-1, keepdim=False)[0] # (B, 128, N)

        # Concatenate multi-scale local features
        local_feat = torch.cat((e1, e2, e3), dim=1) # (B, 256, N)
        local_feat = self.conv4(local_feat)         # (B, 256, N)

        # Global feature aggregation
        global_feat = local_feat.max(dim=-1, keepdim=True)[0] # (B, 256, 1)
        global_feat_expanded = global_feat.repeat(1, 1, num_points) # (B, 256, N)

        # Combine local + global
        combined = torch.cat((local_feat, global_feat_expanded), dim=1) # (B, 512, N)
        desc = self.conv5(combined) # (B, descriptor_dim, N)

        # Permute to (B, N, descriptor_dim) and L2-normalize
        desc = desc.transpose(2, 1).contiguous()
        desc = F.normalize(desc, p=2, dim=-1)

        return desc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing GeometricDGCNN on device:", device)
    model = GeometricDGCNN(k=20, descriptor_dim=32).to(device)
    dummy_input = torch.randn(4, 2048, 3, device=device)
    
    out = model(dummy_input)
    print("Output shape:", out.shape)
    assert out.shape == (4, 2048, 32)
    norms = torch.norm(out, dim=-1)
    print("Mean norm:", norms.mean().item(), "Min norm:", norms.min().item(), "Max norm:", norms.max().item())
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    print("GeometricDGCNN model test PASSED!")
