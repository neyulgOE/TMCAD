import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE with optional Hard Negative Mining.

    Standard mode: uses all K x K negatives in the mini-batch.
    Hard negative mode: additionally mines the top-`n_hard` hardest
    negatives per anchor (the closest non-matching descriptors) and
    up-weights them in the softmax denominator.
    """
    def __init__(self, temperature=0.07, hard_negative_k=64):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_k = hard_negative_k   # number of hardest negatives to mine

    def forward(self, cad_desc, obs_desc, num_pos=512):
        """
        cad_desc: (B, N, D) L2-normalized CAD point descriptors
        obs_desc: (B, N, D) L2-normalized Observed point descriptors
        num_pos:  number of leading points that are ground-truth correspondences
        """
        B = cad_desc.size(0)
        device = cad_desc.device
        targets = torch.arange(num_pos, device=device)

        total_loss = torch.tensor(0.0, device=device)
        total_top1 = torch.tensor(0.0, device=device)
        total_top5 = torch.tensor(0.0, device=device)

        for b in range(B):
            f_cad = cad_desc[b, :num_pos, :]   # (K, D)
            f_obs = obs_desc[b, :num_pos, :]   # (K, D)

            # Full cosine similarity matrix: (K, K)
            sim = torch.matmul(f_cad, f_obs.t()) / self.temperature

            # Hard negative mining: mask out positives, find hardest negatives
            # We re-weight by adding a margin to the hardest negatives
            if self.hard_negative_k > 0 and num_pos > self.hard_negative_k + 1:
                with torch.no_grad():
                    # Build negative mask: True where (i, j) is a negative pair
                    neg_mask = torch.ones(num_pos, num_pos, dtype=torch.bool, device=device)
                    neg_mask.fill_diagonal_(False)

                    # Find hardest negatives for each anchor
                    neg_sim = sim.detach().clone()
                    neg_sim[~neg_mask] = -1e9  # suppress positives

                    # Top-k hardest: highest similarity among non-matching pairs
                    hard_k = min(self.hard_negative_k, num_pos - 1)
                    hard_idx = neg_sim.topk(hard_k, dim=1)[1]   # (K, hard_k)

                    # Build hard-negative weight boost mask
                    boost = torch.zeros_like(sim)
                    boost.scatter_(1, hard_idx, 0.5 / self.temperature)  # additive margin

                # Apply boost: make hard negatives appear "harder" in the logits
                sim_boosted = sim + boost
            else:
                sim_boosted = sim

            # Symmetric cross-entropy loss
            loss_c2o = F.cross_entropy(sim_boosted, targets)
            loss_o2c = F.cross_entropy(sim_boosted.t(), targets)
            loss_b = 0.5 * (loss_c2o + loss_o2c)
            total_loss = total_loss + loss_b

            # Metrics (use un-boosted sim for fair evaluation)
            with torch.no_grad():
                pred1 = sim.argmax(dim=1)
                top1_acc = (pred1 == targets).float().mean()

                k_val = min(5, num_pos)
                top5 = sim.topk(k=k_val, dim=1)[1]
                top5_acc = (top5 == targets.unsqueeze(1)).any(dim=1).float().mean()

                total_top1 = total_top1 + top1_acc
                total_top5 = total_top5 + top5_acc

        return total_loss / B, total_top1 / B, total_top5 / B


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing SymmetricInfoNCELoss on device:", device)

    loss_fn = SymmetricInfoNCELoss(temperature=0.07, hard_negative_k=64)

    B, N, D, K = 4, 2048, 64, 256
    cad_desc = F.normalize(torch.randn(B, N, D, device=device), p=2, dim=-1)
    obs_desc = F.normalize(torch.randn(B, N, D, device=device), p=2, dim=-1)

    loss_rand, top1_rand, top5_rand = loss_fn(cad_desc, obs_desc, num_pos=K)
    print(f"Random - Loss: {loss_rand.item():.4f}, Top-1: {top1_rand.item()*100:.2f}%, Top-5: {top5_rand.item()*100:.2f}%")

    obs_desc_perf = F.normalize(cad_desc.clone() + torch.randn_like(cad_desc) * 0.01, p=2, dim=-1)
    loss_perf, top1_perf, top5_perf = loss_fn(cad_desc, obs_desc_perf, num_pos=K)
    print(f"Perfect - Loss: {loss_perf.item():.4f}, Top-1: {top1_perf.item()*100:.2f}%, Top-5: {top5_perf.item()*100:.2f}%")
    assert top1_perf.item() > 0.95
    print("SymmetricInfoNCELoss test PASSED!")
