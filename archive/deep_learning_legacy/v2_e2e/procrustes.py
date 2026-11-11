import torch
import torch.nn as nn
import torch.nn.functional as F

def differentiable_kabsch(src_pts, dst_pts, weights=None, eps=1e-8):
    """
    Computes optimal rotation R and translation t such that:
        dst_pts approx R @ src_pts + t
    using weighted Differentiable Kabsch SVD (Procrustes analysis).

    Args:
        src_pts: (B, K, 3) Source 3D points (e.g., CAD keypoints)
        dst_pts: (B, K, 3) Target 3D points (e.g., Voted Camera keypoints)
        weights: (B, K) Optional non-negative confidence weights
    Returns:
        R: (B, 3, 3) Rotation matrix in SO(3)
        t: (B, 3) Translation vector
    """
    B, K, _ = src_pts.shape
    device = src_pts.device

    if weights is None:
        weights = torch.ones(B, K, 1, device=device, dtype=src_pts.dtype)
    else:
        if weights.dim() == 2:
            weights = weights.unsqueeze(-1)  # (B, K, 1)
        weights = weights + eps

    w_sum = torch.sum(weights, dim=1, keepdim=True) + eps  # (B, 1, 1)
    norm_weights = weights / w_sum                          # (B, K, 1)

    # Weighted centroids
    centroid_src = torch.sum(src_pts * norm_weights, dim=1, keepdim=True)  # (B, 1, 3)
    centroid_dst = torch.sum(dst_pts * norm_weights, dim=1, keepdim=True)  # (B, 1, 3)

    # Centered coordinates
    src_centered = src_pts - centroid_src  # (B, K, 3)
    dst_centered = dst_pts - centroid_dst  # (B, K, 3)

    # Cross-covariance matrix H: (B, 3, 3)
    # H = src_centered^T @ diag(w) @ dst_centered
    H = torch.bmm((src_centered * norm_weights).transpose(1, 2), dst_centered)  # (B, 3, 3)

    # SVD: H = U @ S @ V^T
    # Note: torch.linalg.svd returns U, S, Vh where Vh = V^T
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)  # (B, 3, 3)

    # Handle reflection case (det(V @ U^T) could be -1)
    # R = V @ diag(1, 1, det(V @ U^T)) @ U^T
    VU_T = torch.bmm(V, U.transpose(1, 2))
    d = torch.det(VU_T)  # (B,)

    eye = torch.eye(3, device=device, dtype=src_pts.dtype).unsqueeze(0).repeat(B, 1, 1)
    eye[:, 2, 2] = d

    R = torch.bmm(torch.bmm(V, eye), U.transpose(1, 2))  # (B, 3, 3)

    # Translation: t = centroid_dst - R @ centroid_src
    t = centroid_dst.squeeze(1) - torch.bmm(R, centroid_src.transpose(1, 2)).squeeze(-1)  # (B, 3)

    return R, t


def geodesic_rotation_loss(R_pred, R_gt, eps=1e-6):
    """
    Computes geodesic distance on SO(3) between predicted and ground-truth rotation matrices.
    loss = arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    """
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)  # (B, 3, 3)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # radians
    return theta.mean()


def add_loss(R_pred, t_pred, R_gt, t_gt, cad_points):
    """
    Average Distance of Model Points (ADD) Loss:
    Computes mean L1 distance between CAD points transformed by (R_pred, t_pred)
    and (R_gt, t_gt).
    cad_points: (B, N, 3) or (N, 3)
    """
    if cad_points.dim() == 2:
        cad_points = cad_points.unsqueeze(0).repeat(R_pred.size(0), 1, 1)  # (B, N, 3)

    # (B, N, 3) @ (B, 3, 3)^T + (B, 1, 3)
    pts_pred = torch.bmm(cad_points, R_pred.transpose(1, 2)) + t_pred.unsqueeze(1)
    pts_gt   = torch.bmm(cad_points, R_gt.transpose(1, 2))   + t_gt.unsqueeze(1)

    loss = F.l1_loss(pts_pred, pts_gt)
    return loss


class EndToEndPoseLoss(nn.Module):
    def __init__(self, w_add=1.0, w_trans=5.0, w_rot=2.0, w_kp=10.0):
        super().__init__()
        self.w_add = w_add
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.w_kp = w_kp

    def forward(self, R_pred, t_pred, kp_pred, R_gt, t_gt, kp_gt, cad_points):
        l_add = add_loss(R_pred, t_pred, R_gt, t_gt, cad_points)
        l_trans = F.l1_loss(t_pred, t_gt)
        l_rot = geodesic_rotation_loss(R_pred, R_gt)
        l_kp = F.smooth_l1_loss(kp_pred, kp_gt)

        total_loss = (self.w_add * l_add +
                      self.w_trans * l_trans +
                      self.w_rot * l_rot +
                      self.w_kp * l_kp)

        return total_loss, {
            "loss_add": l_add.item(),
            "loss_trans": l_trans.item(),
            "loss_rot": l_rot.item(),
            "loss_kp": l_kp.item(),
            "total_loss": total_loss.item()
        }


if __name__ == "__main__":
    B, K = 4, 8
    # Source points (e.g. CAD keypoints)
    src = torch.randn(B, K, 3, requires_grad=True)

    # Create random GT transformation
    from scipy.spatial.transform import Rotation
    R_gt_np = Rotation.random(B).as_matrix().astype("float32")
    t_gt_np = torch.randn(B, 3).numpy()

    R_gt = torch.from_numpy(R_gt_np)
    t_gt = torch.from_numpy(t_gt_np)

    # Target points: dst = R @ src + t + small noise
    dst = torch.bmm(src, R_gt.transpose(1, 2)) + t_gt.unsqueeze(1)
    dst = dst + 0.001 * torch.randn_like(dst)

    R_pred, t_pred = differentiable_kabsch(src, dst)
    print("R_pred shape:", R_pred.shape, "t_pred shape:", t_pred.shape)

    loss_fn = EndToEndPoseLoss()
    cad_pts = torch.randn(B, 500, 3)
    loss, metrics = loss_fn(R_pred, t_pred, dst, R_gt, t_gt, dst, cad_pts)
    print("Initial Loss metrics:", metrics)

    loss.backward()
    print("src grad norm:", src.grad.norm().item())
    assert src.grad is not None
    print("Differentiable Kabsch & PoseLoss tests PASSED!")
