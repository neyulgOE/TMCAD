import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetryAwareInfoNCELoss(nn.Module):
    """
    Symmetry-Aware InfoNCE Multi-Positive Contrastive Loss.
    
    Given:
      - similarity_matrix S: shape (B, N_obs, N_cad)
      - positive_targets P: shape (B, N_obs, num_sym) containing the CAD key primitive indices
        corresponding to all physically symmetric variations (e.g. 0 deg and 180 deg for C2_Z).
        
    Formula:
      L_i = logsumexp_{all j}(S_{i, j}) - logsumexp_{j in P_i}(S_{i, j})
      
    When num_sym == 1 (asymmetric), this is strictly equivalent to standard multi-class cross-entropy.
    When num_sym > 1, any of the symmetry-equivalent CAD primitives is recognized as a valid match,
    eliminating ambiguous symmetry penalty gradients.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, similarity_matrix: torch.Tensor, positive_targets: torch.Tensor) -> torch.Tensor:
        """
        similarity_matrix: (B, N_obs, N_cad)
        positive_targets: (B, N_obs, num_sym)
        """
        B, N_obs, N_cad = similarity_matrix.shape
        num_sym = positive_targets.shape[-1]

        # Logsumexp over all negative + positive candidates (denominator)
        denom_logsumexp = torch.logsumexp(similarity_matrix, dim=-1) # (B, N_obs)

        # Gather positive logits: shape (B, N_obs, num_sym)
        # positive_targets holds indices in [0, N_cad - 1]
        pos_logits = torch.gather(similarity_matrix, dim=-1, index=positive_targets)

        # Logsumexp over the positive symmetric candidates (numerator)
        numer_logsumexp = torch.logsumexp(pos_logits, dim=-1) # (B, N_obs)

        # Multi-positive InfoNCE loss
        loss = denom_logsumexp - numer_logsumexp # (B, N_obs)
        return torch.mean(loss)

def compute_matching_recall(similarity_matrix: torch.Tensor, positive_targets: torch.Tensor, topk: int = 1) -> float:
    """
    Computes matching recall@topk: fraction of observation points whose top-k
    predicted CAD primitive matches ANY of the symmetric ground truth targets.
    """
    with torch.no_grad():
        # Top-k predictions: (B, N_obs, topk)
        _, topk_preds = torch.topk(similarity_matrix, k=topk, dim=-1)

        # Check if any topk_pred is in positive_targets
        # topk_preds: (B, N_obs, topk, 1)
        # positive_targets: (B, N_obs, 1, num_sym)
        preds = topk_preds.unsqueeze(-1)
        targets = positive_targets.unsqueeze(2)

        # matches: (B, N_obs, topk, num_sym)
        matches = (preds == targets)
        # Any match along topk and num_sym dimensions
        hit = torch.any(torch.any(matches, dim=-1), dim=-1).float() # (B, N_obs)
        return float(torch.mean(hit).item() * 100.0)


class TriangleAwareHierarchyLoss(nn.Module):
    """
    Hàm Loss Kết Hợp Cho TMCAD Single-CAD Hierarchy:
    1. Symmetry-Aware Multi-Positive InfoNCE Loss (Xử lý đối xứng C2, C4, D2).
    2. Differentiable Point-to-Triangle-Plane Loss (Tận dụng trực tiếp tâm & pháp tuyến tam giác CAD).
    3. Occlusion/Outlier Confidence Supervision (Giúp lọc bỏ vùng che khuất & nhiễu nền khi giải SVD).
    """
    def __init__(
        self,
        temperature: float = 0.07,
        pos_radius_m: float = 0.035,
        plane_loss_weight: float = 0.50,
        coord_loss_weight: float = 0.35,
        conf_loss_weight: float = 0.20,
    ):
        super().__init__()
        self.temperature = temperature
        self.pos_radius_m = pos_radius_m
        self.plane_loss_weight = plane_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.conf_loss_weight = conf_loss_weight

    def forward(
        self,
        sim_matrix: torch.Tensor,
        gt_cad_points: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_normals: torch.Tensor,
        symmetry_matrices: torch.Tensor,
        inlier_mask: torch.Tensor = None,
        pred_confidence: torch.Tensor = None,
    ):
        B, N, M = sim_matrix.shape
        device = sim_matrix.device

        if patch_centers.dim() == 2:
            patch_centers = patch_centers.unsqueeze(0).expand(B, -1, -1)
        if patch_normals.dim() == 2:
            patch_normals = patch_normals.unsqueeze(0).expand(B, -1, -1)
        if symmetry_matrices.dim() == 4:
            sym_mats = symmetry_matrices[0]
        else:
            sym_mats = symmetry_matrices

        if inlier_mask is None:
            inlier_mask = torch.ones((B, N), dtype=torch.float32, device=device)

        # 1. Tính khoảng cách đối xứng nhỏ nhất giữa mỗi điểm GT và tất cả các Patch tam giác
        gt_sym = torch.einsum("sij,bnj->bsni", sym_mats, gt_cad_points)
        dists_sym = torch.cdist(
            gt_sym.reshape(B * sym_mats.shape[0], N, 3),
            patch_centers.repeat_interleave(sym_mats.shape[0], dim=0),
            p=2.0,
        ).reshape(B, sym_mats.shape[0], N, M)

        min_dists, best_sym_idx = torch.min(dists_sym, dim=1)  # (B, N, M)

        # Mặt nạ các patch dương (Positive Patches) trong bán kính pos_radius_m
        pos_mask = min_dists <= self.pos_radius_m
        nearest_patch_idx = torch.argmin(min_dists, dim=-1, keepdim=True)  # (B, N, 1)
        pos_mask.scatter_(-1, nearest_patch_idx, True)

        # 2. Symmetry-Aware Multi-Positive InfoNCE Loss
        logits = sim_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(logits - logits_max.detach())

        pos_exp_sum = torch.sum(exp_logits * pos_mask.float(), dim=-1)
        all_exp_sum = torch.sum(exp_logits, dim=-1) + 1e-8
        per_point_infonce = -torch.log((pos_exp_sum + 1e-8) / all_exp_sum)

        valid_weights = inlier_mask / (torch.sum(inlier_mask, dim=-1, keepdim=True) + 1e-6)
        loss_infonce = torch.sum(per_point_infonce * valid_weights) / B

        # 3. Differentiable Soft Point-to-Triangle-Plane Loss
        assign_probs = F.softmax(logits, dim=-1)  # (B, N, M)
        pred_cad_centers = torch.bmm(assign_probs, patch_centers)  # (B, N, 3)
        pred_cad_normals = F.normalize(torch.bmm(assign_probs, patch_normals), p=2.0, dim=-1)  # (B, N, 3)

        nearest_flat = nearest_patch_idx.squeeze(-1)  # (B, N)
        sym_choice = torch.gather(best_sym_idx, dim=-1, index=nearest_patch_idx).squeeze(-1)  # (B, N)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        point_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        gt_aligned = gt_sym[batch_idx, sym_choice, point_idx]  # (B, N, 3)

        diff_vec = gt_aligned - pred_cad_centers
        plane_residual = torch.abs(torch.sum(pred_cad_normals * diff_vec, dim=-1))
        coord_residual = torch.norm(diff_vec, p=2.0, dim=-1)

        loss_plane = torch.sum(plane_residual * valid_weights) / B
        loss_coord = torch.sum(coord_residual * valid_weights) / B

        # 4. Inlier Confidence Loss
        if pred_confidence is None:
            max_sim, _ = torch.max(sim_matrix, dim=-1)
            pred_confidence = torch.sigmoid(max_sim * 5.0)
        loss_conf = F.binary_cross_entropy(
            torch.clamp(pred_confidence, 1e-5, 1.0 - 1e-5), inlier_mask
        )

        total_loss = (
            loss_infonce
            + self.plane_loss_weight * loss_plane
            + self.coord_loss_weight * loss_coord
            + self.conf_loss_weight * loss_conf
        )

        with torch.no_grad():
            top5_idx = torch.topk(sim_matrix, k=min(5, M), dim=-1).indices
            top1_idx = top5_idx[:, :, :1]
            r1_hits = torch.gather(pos_mask, dim=-1, index=top1_idx).squeeze(-1).float()
            r5_hits = torch.any(torch.gather(pos_mask, dim=-1, index=top5_idx), dim=-1).float()
            recall_1 = float(torch.sum(r1_hits * valid_weights) / B)
            recall_5 = float(torch.sum(r5_hits * valid_weights) / B)

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_infonce": float(loss_infonce.item()),
            "loss_plane": float(loss_plane.item()),
            "loss_coord": float(loss_coord.item()),
            "recall_at_1": recall_1,
            "recall_at_5": recall_5,
        }
        return total_loss, metrics


class TriangleAwareTopologyLoss(nn.Module):
    """
    Hàm Loss tổng hợp cho Single-CAD On-the-fly Training:
      1. Symmetry-Aware Multi-Positive InfoNCE (chỉ tính trên điểm inlier)
      2. Differentiable Soft Point-to-Triangle-Plane Loss (L_plane)
      3. Topological Region Smoothness Loss trên k=4 láng giềng 3D (L_topo)
      4. Inlier Confidence BCE Loss (L_conf)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_plane: float = 2.0,
        lambda_topo: float = 1.0,
        lambda_conf: float = 0.5,
        topo_k: int = 4,
        pos_radius_ratio: float = 0.06,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_plane = float(lambda_plane)
        self.lambda_topo = float(lambda_topo)
        self.lambda_conf = float(lambda_conf)
        self.topo_k = int(topo_k)
        self.pos_radius_ratio = float(pos_radius_ratio)

    def forward(
        self,
        obs_desc: torch.Tensor,            # (B, N, D) đã L2-normalize
        cad_desc: torch.Tensor,            # (B, M, D) đã L2-normalize
        patch_centers: torch.Tensor,       # (B, M, 3)
        patch_normals: torch.Tensor,       # (B, M, 3)
        obs_points_cad: torch.Tensor,      # (B, N, 3) ground-truth CAD coords
        obs_points_cam: torch.Tensor,      # (B, N, 3) camera coords (cho k-NN 3D)
        obs_patch_targets: torch.Tensor,   # (B, N) chỉ số patch gần nhất (-1 cho clutter)
        obs_region_ids: torch.Tensor,      # (B, N) region_id từ _regions.npz (-1 cho clutter)
        inlier_mask: torch.Tensor,         # (B, N) 1.0 vật thể, 0.0 clutter
        conf_logits: torch.Tensor = None,  # (B, N) đầu ra từ inlier_head (tùy chọn)
        sym_matrices: torch.Tensor = None, # (B, S, 3, 3) hoặc (S, 3, 3)
        cad_diag: float = 1.0,
    ):
        B, N, _ = obs_desc.shape
        M = cad_desc.shape[1]
        device = obs_desc.device
        diag = max(1e-4, float(cad_diag))

        if sym_matrices is not None and sym_matrices.dim() == 4:
            sym_mats = sym_matrices[0]
        else:
            sym_mats = sym_matrices

        sim = torch.bmm(obs_desc, cad_desc.transpose(1, 2))               # (B, N, M)
        logits = sim / self.temperature                                   # (B, N, M)
        probs = torch.softmax(logits, dim=-1)                             # (B, N, M)

        valid_mask = (inlier_mask > 0.5) & (obs_patch_targets >= 0)       # (B, N)
        num_valid = valid_mask.sum().clamp(min=1)

        pos_radius = self.pos_radius_ratio * diag
        with torch.no_grad():
            if sym_mats is not None and sym_mats.numel() > 0:
                pos_mask = torch.zeros((B, N, M), dtype=torch.bool, device=device)
                for s_idx in range(sym_mats.shape[0]):
                    S_mat = sym_mats[s_idx].to(device=device, dtype=obs_points_cad.dtype)
                    obs_sym = torch.matmul(obs_points_cad, S_mat.t())
                    d_sym = torch.cdist(obs_sym, patch_centers)
                    pos_mask |= (d_sym <= pos_radius)
            else:
                d_mat = torch.cdist(obs_points_cad, patch_centers)
                pos_mask = d_mat <= pos_radius

            safe_targets = obs_patch_targets.clamp(min=0, max=M - 1)
            pos_mask.scatter_(2, safe_targets.unsqueeze(-1), True)

        log_prob = F.log_softmax(logits, dim=-1)
        pos_float = pos_mask.float()
        pos_count = pos_float.sum(dim=-1).clamp(min=1.0)
        neg_inf = torch.full_like(logits, -1e9)
        pos_log_prob = torch.logsumexp(torch.where(pos_mask, log_prob, neg_inf), dim=-1) - torch.log(pos_count)
        loss_infonce = -(pos_log_prob * valid_mask.float()).sum() / num_valid

        pred_c = torch.bmm(probs, patch_centers)
        pred_n = F.normalize(torch.bmm(probs, patch_normals), dim=-1, eps=1e-6)
        diff_cad = obs_points_cad - pred_c
        plane_dist = torch.abs((diff_cad * pred_n).sum(dim=-1)) / diag
        point_dist = torch.norm(diff_cad, dim=-1) / diag
        loss_plane = ((plane_dist + 0.5 * point_dist) * valid_mask.float()).sum() / num_valid

        loss_topo = torch.tensor(0.0, device=device)
        if self.lambda_topo > 0.0 and N > self.topo_k + 1:
            with torch.no_grad():
                cam_dists = torch.cdist(obs_points_cam, obs_points_cam)
                knn_idx = torch.topk(cam_dists, k=self.topo_k + 1, dim=-1, largest=False).indices[:, :, 1:]

            batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, N, self.topo_k)
            nb_pred_c = pred_c[batch_idx, knn_idx]
            nb_reg = obs_region_ids[batch_idx, knn_idx]
            nb_valid = valid_mask[batch_idx, knn_idx]

            same_region = (
                (nb_reg == obs_region_ids.unsqueeze(-1))
                & (obs_region_ids.unsqueeze(-1) >= 0)
                & valid_mask.unsqueeze(-1)
                & nb_valid
            )
            num_pairs = same_region.sum()
            if num_pairs > 0:
                c_jump = torch.norm(pred_c.unsqueeze(2) - nb_pred_c, dim=-1) / diag
                cam_gap = cam_dists.gather(2, knn_idx) / diag
                excess_jump = F.relu(c_jump - 1.5 * cam_gap)
                loss_topo = (excess_jump * same_region.float()).sum() / num_pairs.clamp(min=1)

        loss_conf = torch.tensor(0.0, device=device)
        if conf_logits is not None and self.lambda_conf > 0.0:
            loss_conf = F.binary_cross_entropy_with_logits(conf_logits, inlier_mask.float())

        total_loss = (
            loss_infonce
            + self.lambda_plane * loss_plane
            + self.lambda_topo * loss_topo
            + self.lambda_conf * loss_conf
        )

        with torch.no_grad():
            top5 = torch.topk(sim, k=min(5, M), dim=-1).indices
            hit_top5 = pos_mask.gather(2, top5)
            recall_1 = float(((hit_top5[:, :, 0] & valid_mask).float().sum() / num_valid).item() * 100.0)
            recall_5 = float(((hit_top5.any(dim=-1) & valid_mask).float().sum() / num_valid).item() * 100.0)

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_infonce": float(loss_infonce.detach().item()),
            "loss_plane": float(loss_plane.detach().item()),
            "loss_topo": float(loss_topo.detach().item()),
            "loss_conf": float(loss_conf.detach().item()),
            "recall_at_1": recall_1,
            "recall_at_5": recall_5,
        }
        return total_loss, metrics


