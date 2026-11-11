#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hàm Loss Kết Hợp Cho TMCAD Single-CAD Hierarchy:
1. Symmetry-Aware Multi-Positive InfoNCE Loss (Xử lý đối xứng C2, C4, D2).
2. Differentiable Point-to-Triangle-Plane Loss (Tận dụng trực tiếp tâm & pháp tuyến tam giác CAD).
3. Occlusion/Outlier Confidence Supervision (Giúp lọc bỏ vùng che khuất & nhiễu nền khi giải SVD).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangleAwareHierarchyLoss(nn.Module):
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
        inlier_mask: Optional[torch.Tensor] = None,
        pred_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            sim_matrix: (B, N, M) ma trận tương đồng cosine giữa N điểm quan sát và M patch tam giác CAD.
            gt_cad_points: (B, N, 3) tọa độ 3D chuẩn trên hệ quy chiếu CAD.
            patch_centers: (B, M, 3) hoặc (M, 3) tâm của từng LocalPatchBundle.
            patch_normals: (B, M, 3) hoặc (M, 3) vector pháp tuyến chuẩn của từng Patch tam giác.
            symmetry_matrices: (B, S, 3, 3) hoặc (S, 3, 3) nhóm ma trận đối xứng G = {S_k}.
            inlier_mask: (B, N) nhãn điểm hợp lệ (1.0 = bề mặt thật, 0.0 = nhiễu nền/che khuất).
            pred_confidence: (B, N) độ tin cậy dự đoán của từng điểm (nếu có).
        """
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
        # gt_sym: (B, S, N, 3)
        gt_sym = torch.einsum("sij,bnj->bsni", sym_mats, gt_cad_points)
        # dists_sym: (B, S, N, M)
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
        # Tính tọa độ tâm tam giác và pháp tuyến kỳ vọng từ phân phối xác suất softmax
        assign_probs = F.softmax(logits, dim=-1)  # (B, N, M)
        pred_cad_centers = torch.bmm(assign_probs, patch_centers)  # (B, N, 3)
        pred_cad_normals = F.normalize(torch.bmm(assign_probs, patch_normals), p=2.0, dim=-1)  # (B, N, 3)

        # Chọn phép biến đổi đối xứng khớp nhất cho từng điểm để tính khoảng cách hình học
        nearest_flat = nearest_patch_idx.squeeze(-1)  # (B, N)
        sym_choice = torch.gather(best_sym_idx, dim=-1, index=nearest_patch_idx).squeeze(-1)  # (B, N)
        # Lấy tọa độ GT đã xoay theo nhánh đối xứng tối ưu
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        point_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        gt_aligned = gt_sym[batch_idx, sym_choice, point_idx]  # (B, N, 3)

        diff_vec = gt_aligned - pred_cad_centers
        # Khoảng cách vuông góc tới mặt phẳng chứa tam giác: |n^T (p_gt - c_pred)|
        plane_residual = torch.abs(torch.sum(pred_cad_normals * diff_vec, dim=-1))
        # Khoảng cách Euclid mềm tới tâm tam giác
        coord_residual = torch.norm(diff_vec, p=2.0, dim=-1)

        loss_plane = torch.sum(plane_residual * valid_weights) / B
        loss_coord = torch.sum(coord_residual * valid_weights) / B

        # 4. Inlier Confidence Loss (nếu mạng có đầu ra dự đoán độ tin cậy hoặc dùng max similarity)
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

        # Tính chỉ số Recall@1 và Recall@5 có xét đối xứng
        with torch.no_grad():
            top5_idx = torch.topk(sim_matrix, k=min(5, M), dim=-1).indices  # (B, N, 5)
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