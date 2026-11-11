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
