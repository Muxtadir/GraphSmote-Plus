"""
Topology-Aware Margin (TAM) helpers for GraphSMOTE-Plus.

Faithful reimplementation of Song, Park, Yang (ICML 2022) ACM/ADM math from the
official repo https://github.com/Jaeyun-Song/TAM (models/tam.py), using pure
PyTorch mean aggregation instead of torch_geometric MessagePassing — no new deps.

Attribution: TAM = Song et al., ICML 2022. GraphSMOTE = Zhao et al., WSDM 2021.
This module is a controlled hybrid hook; it does not claim inventing TAM.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_jsd(dist1, dist2):
    """Jensen–Shannon divergence (elementwise), matching official TAM."""
    dist_mean = (dist1 + dist2) / 2.0
    jsd = (
        F.kl_div(dist_mean.log(), dist1, reduction="none")
        + F.kl_div(dist_mean.log(), dist2, reduction="none")
    ) / 2.0
    return jsd


def mean_aggregate(x, adj):
    """
    Mean aggregation with self-loops (official MeanAggregation + add_self_loops).
    adj: dense [N, N] (nonnegative); self-loops added if missing on diagonal.
    """
    if adj.is_sparse:
        adj = adj.to_dense()
    adj = adj.clone()
    n = adj.shape[0]
    idx = torch.arange(n, device=adj.device)
    adj[idx, idx] = torch.clamp(adj[idx, idx], min=1.0)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj @ x / deg


@torch.no_grad()
def compute_tam(
    output,
    adj,
    label,
    train_mask,
    class_num_list,
    temp_phi=1.2,
    temp_gamma=0.4,
):
    """
    Compute ACM and ADM for labeled train nodes.

    Mirrors official compute_tam (detached). train_mask: bool [N] over original graph.
    Returns acm, adm each of shape [#train_labeled, #classes] aligned with train_mask.nonzero order
    — actually official indexes by train_mask directly on neighbor_dist which is already
    restricted to train_mask rows. We return full [N, C] tensors with meaningful rows on train_mask.
    """
    n_cls = int(label.max().item()) + 1
    device = output.device

    cls_num_list = torch.as_tensor(class_num_list, dtype=torch.float, device=device)
    cls_num_ratio = cls_num_list / cls_num_list.sum().clamp(min=1.0)
    cls_num_ratio = cls_num_ratio * temp_gamma + (1.0 - temp_gamma)
    max_beta = torch.max(cls_num_ratio)
    cls_temperature = (temp_phi * (cls_num_ratio + 1.0 - max_beta)).unsqueeze(0)
    temp = 1.0 / cls_temperature

    # Predict unlabeled; overwrite train labels with one-hot (no val/test gold labels).
    agg_out = F.softmax(output.clone().detach() / temp, dim=1)
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    agg_out[train_idx] = F.one_hot(label[train_idx].clone(), num_classes=n_cls).float()

    neighbor_full = mean_aggregate(agg_out, adj)
    neighbor_dist = neighbor_full[train_mask]  # [#labeled, C]

    connectivity_matrix = []
    train_labels = label[train_mask]
    for c in range(n_cls):
        c_mask = train_labels == c
        if c_mask.any():
            connectivity_matrix.append(neighbor_dist[c_mask].mean(dim=0))
        else:
            # Empty class (should be rare): uniform fallback for numerical stability.
            connectivity_matrix.append(
                torch.full((n_cls,), 1.0 / n_cls, device=device, dtype=neighbor_dist.dtype)
            )
    connectivity_matrix = torch.stack(connectivity_matrix, dim=0)

    center_mask = F.one_hot(train_labels.clone(), num_classes=n_cls).bool()
    neighbor_dist = neighbor_dist.clone()
    neighbor_dist[neighbor_dist < 1e-6] = 1e-6
    connectivity_matrix = connectivity_matrix.clone()
    connectivity_matrix[connectivity_matrix < 1e-6] = 1e-6

    # ACM
    acm = (
        (neighbor_dist[center_mask].unsqueeze(dim=1) / torch.diagonal(connectivity_matrix).unsqueeze(dim=1)[train_labels])
        * (connectivity_matrix[train_labels] / neighbor_dist)
    )
    acm[acm > 1] = 1
    acm[center_mask] = 1

    # ADM
    cls_pair_jsd = compute_jsd(
        connectivity_matrix.unsqueeze(dim=0), connectivity_matrix.unsqueeze(dim=1)
    ).sum(dim=-1)
    cls_pair_jsd[cls_pair_jsd < 1e-6] = 1e-6
    self_kl = compute_jsd(neighbor_dist, connectivity_matrix[train_labels]).sum(dim=-1, keepdim=True)
    neighbor_kl = compute_jsd(
        neighbor_dist.unsqueeze(1), connectivity_matrix.unsqueeze(0)
    ).sum(dim=-1)
    adm = (self_kl ** 2 + (cls_pair_jsd ** 2)[train_labels] - neighbor_kl ** 2) / (
        2 * (cls_pair_jsd ** 2)[train_labels]
    )
    adm[center_mask] = 0

    return acm, adm


def adjust_logits_tam(
    output,
    adj,
    labels,
    idx_train,
    class_num_list,
    epoch,
    tam_alpha=2.5,
    tam_beta=0.5,
    tam_phi=1.2,
    tam_delta=0.4,
    tam_warmup=5,
    tam_scope="train_orig",
    idx_train_new=None,
    n_orig=None,
):
    """
    Return logits for CE. Unadjusted copy unless TAM active.

    Official activates when epoch > warmup with 1-based epochs; we use (epoch+1) > warmup.
    Margins applied only on scope nodes; other logits unchanged (val/test use raw `output` elsewhere).
    """
    # Official loop is 1-based and activates when epoch > warmup.
    # Map our 0-based epoch via (epoch + 1) so warmup=5 ⇒ first TAM epoch index 5.
    if (epoch + 1) <= tam_warmup:
        return output

    n_orig = int(n_orig if n_orig is not None else labels.shape[0])
    # Topology always on original graph block.
    if adj.dim() == 2 and adj.shape[0] > n_orig:
        adj_topo = adj[:n_orig, :n_orig]
        logits_topo = output[:n_orig]
        labels_topo = labels[:n_orig]
    else:
        adj_topo = adj
        logits_topo = output[:n_orig] if output.shape[0] >= n_orig else output
        labels_topo = labels[:n_orig] if labels.shape[0] >= n_orig else labels

    train_mask = torch.zeros(n_orig, dtype=torch.bool, device=output.device)
    # Only original train indices (clamp to graph)
    orig_train = idx_train[(idx_train < n_orig)]
    train_mask[orig_train] = True

    acm, adm = compute_tam(
        logits_topo,
        adj_topo,
        labels_topo,
        train_mask,
        class_num_list=class_num_list,
        temp_phi=tam_phi,
        temp_gamma=tam_delta,
    )

    # Map train_mask rows → margin matrices
    m_acm = acm.log()
    m_adm = -adm

    adjusted = output.clone()
    train_idx_local = train_mask.nonzero(as_tuple=False).view(-1)

    if tam_scope == "train_orig":
        target_idx = train_idx_local
        # acm/adm rows align with train_mask order
        adjusted[target_idx] = adjusted[target_idx] + tam_alpha * m_acm + tam_beta * m_adm
    elif tam_scope == "train_all":
        # Apply to original train; synthetics keep unadjusted logits (topology is original-only in v1).
        adjusted[train_idx_local] = (
            adjusted[train_idx_local] + tam_alpha * m_acm + tam_beta * m_adm
        )
    else:
        raise ValueError("Unknown tam_scope: {}".format(tam_scope))

    # One-line confirmation that TAM margins were applied (does not change math).
    delta = (adjusted[train_idx_local] - output[train_idx_local]).abs().mean().item()
    print("[TAM] active epoch={} scope={} train_nodes={} mean_|delta_logit|={:.6f}".format(
        epoch, tam_scope, int(train_idx_local.numel()), delta))

    return adjusted


def train_class_counts(labels, idx_train, n_cls=None):
    """Per-class counts on train indices only (no val/test)."""
    if n_cls is None:
        n_cls = int(labels.max().item()) + 1
    counts = []
    for c in range(n_cls):
        counts.append(int(((labels[idx_train] == c).sum()).item()))
    return counts
