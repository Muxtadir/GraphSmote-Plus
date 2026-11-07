#!/usr/bin/env python3
"""Unit tests for GraphSMOTE+TAM helpers (no GPU / no full training)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tam as tam_lib  # noqa: E402


def test_mean_aggregate_self_loop():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    adj = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    out = tam_lib.mean_aggregate(x, adj)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_compute_tam_shapes_and_detach():
    n, c = 8, 3
    torch.manual_seed(0)
    logits = torch.randn(n, c, requires_grad=True)
    adj = (torch.rand(n, n) > 0.7).float()
    adj = ((adj + adj.t()) > 0).float()
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    train_mask = torch.tensor([True, True, True, True, False, False, False, False])
    counts = [2, 1, 1]
    acm, adm = tam_lib.compute_tam(logits, adj, labels, train_mask, counts, temp_phi=1.2, temp_gamma=0.4)
    assert acm.shape[0] == int(train_mask.sum())
    assert acm.shape[1] == c
    assert adm.shape == acm.shape
    assert torch.isfinite(acm).all() and torch.isfinite(adm).all()
    # detached: no grad path into acm
    assert acm.requires_grad is False
    assert adm.requires_grad is False


def test_adjust_warmup_noop():
    n, c = 6, 2
    output = torch.randn(n, c)
    adj = torch.eye(n)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    idx_train = torch.tensor([0, 1, 2, 3])
    out = tam_lib.adjust_logits_tam(
        output, adj, labels, idx_train, class_num_list=[2, 2],
        epoch=0, tam_warmup=5,  # epoch+1=1 <= 5 → noop
    )
    assert torch.allclose(out, output)


def test_adjust_applies_after_warmup():
    n, c = 6, 2
    torch.manual_seed(1)
    output = torch.randn(n, c)
    adj = (torch.rand(n, n) > 0.5).float()
    adj = ((adj + adj.t()) > 0).float()
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    idx_train = torch.tensor([0, 1, 2, 3])
    out = tam_lib.adjust_logits_tam(
        output, adj, labels, idx_train, class_num_list=[2, 2],
        epoch=10, tam_warmup=5, tam_alpha=2.5, tam_beta=0.5,
    )
    # Train rows should change; a non-train row should match
    assert not torch.allclose(out[idx_train], output[idx_train])
    assert torch.allclose(out[4], output[4])


def test_train_class_counts():
    labels = torch.tensor([0, 0, 1, 2, 2, 2])
    idx = torch.tensor([0, 2, 3, 4])
    assert tam_lib.train_class_counts(labels, idx, n_cls=3) == [1, 1, 2]


def test_jsd_symmetric_nonneg():
    p = F.softmax(torch.randn(4, 3), dim=-1)
    q = F.softmax(torch.randn(4, 3), dim=-1)
    j = tam_lib.compute_jsd(p, q)
    assert torch.isfinite(j).all()
    assert (j >= -1e-5).all()


if __name__ == "__main__":
    test_mean_aggregate_self_loop()
    test_compute_tam_shapes_and_detach()
    test_adjust_warmup_noop()
    test_adjust_applies_after_warmup()
    test_train_class_counts()
    test_jsd_symmetric_nonneg()
    print("ALL_TAM_TESTS_PASSED")
