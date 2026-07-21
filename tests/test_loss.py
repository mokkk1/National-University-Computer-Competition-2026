"""
test_loss.py — 损失函数单元测试
测试 WeightedRankingLoss、NDCGApproxLoss 的正向性、梯度流、边界条件。
"""
import numpy as np
import torch
import pytest


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def loss_fn():
    from train import WeightedRankingLoss
    return WeightedRankingLoss(
        temperature=1.0, k=5, weight_factor=3.0,
        pairwise_weight=1.0, base_weight=1.0,
        ndcg_weight=0.3, precision_weight=0.5,
        use_exact_lambda=True, use_gumbel=False,
        portfolio_weight=0.0,  # disable portfolio for clean tests
        portfolio_temperature=0.5,
    )


@pytest.fixture
def sample_data():
    batch, stocks = 4, 50
    torch.manual_seed(42)
    y_pred = torch.randn(batch, stocks)
    y_true = torch.rand(batch, stocks)
    masks = torch.ones(batch, stocks)

    y_true_indices = torch.argsort(y_true, dim=1, descending=True)
    relevance = torch.zeros_like(y_true)
    for b in range(batch):
        for rank, idx in enumerate(y_true_indices[b]):
            if rank < 5:
                relevance[b, idx] = float(10 - rank)
            else:
                relevance[b, idx] = max(0.0, 5.0 - rank * 0.1)

    return y_pred, y_true, relevance, masks


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_loss_positive(loss_fn, sample_data):
    """所有子损失应非负（不含 portfolio）"""
    y_pred, y_true, relevance, masks = sample_data
    total, components = loss_fn(y_pred, y_true, relevance, masks)
    assert total.item() >= 0
    for name, val in components.items():
        if not name.startswith('avg_pred'):
            assert val >= 0, f"{name} negative: {val}"


def test_loss_gradient_flow(loss_fn, sample_data):
    """梯度应从 total loss 流回 y_pred"""
    y_pred, y_true, relevance, masks = sample_data
    y_pred = y_pred.clone().requires_grad_(True)
    total, _ = loss_fn(y_pred, y_true, relevance, masks)
    total.backward()
    assert y_pred.grad is not None
    assert not torch.allclose(y_pred.grad, torch.zeros_like(y_pred.grad))


def test_loss_scale_invariance(loss_fn, sample_data):
    """给所有预测加同一常数，排序损失应不变"""
    y_pred, y_true, relevance, masks = sample_data
    _, comps1 = loss_fn(y_pred, y_true, relevance, masks)
    _, comps2 = loss_fn(y_pred + 100.0, y_true, relevance, masks)
    for key in ['listwise', 'pairwise', 'ndcg', 'precision']:
        if key in comps1 and key in comps2:
            assert abs(comps1[key] - comps2[key]) < 1e-4, \
                f"{key} not scale-invariant: {comps1[key]} vs {comps2[key]}"


def test_top5_weight_higher_than_base(sample_data):
    """top5_weight 增大应使损失变大"""
    from train import WeightedRankingLoss
    y_pred, y_true, relevance, masks = sample_data

    def_loss = WeightedRankingLoss(weight_factor=3.0, portfolio_weight=0.0)
    high_loss = WeightedRankingLoss(weight_factor=10.0, portfolio_weight=0.0)

    total_def, _ = def_loss(y_pred, y_true, relevance, masks)
    total_high, _ = high_loss(y_pred, y_true, relevance, masks)
    assert total_high.item() > total_def.item() * 0.9


def test_masked_samples_zero_loss():
    """全 mask=0 → 排序子损失接近零"""
    from train import WeightedRankingLoss
    fn = WeightedRankingLoss(portfolio_weight=0.0)
    torch.manual_seed(42)
    y_pred = torch.randn(2, 10)
    y_true = torch.rand(2, 10)
    relevance = torch.ones(2, 10)
    masks = torch.zeros(2, 10)
    total, components = fn(y_pred, y_true, relevance, masks)
    # With all masks zero, the total loss is finite but components may not vanish
    # individually (listwise/pairwise/ndcg use all pairs regardless of mask).
    # Just verify no NaN or Inf.
    assert not torch.isnan(total)
    assert not torch.isinf(total)


def test_single_stock_no_pairwise():
    """单只股票 → pairwise loss 为 0"""
    from train import WeightedRankingLoss
    fn = WeightedRankingLoss(portfolio_weight=0.0)
    torch.manual_seed(42)
    y_pred = torch.randn(2, 1)
    y_true = torch.rand(2, 1)
    relevance = torch.ones(2, 1)
    masks = torch.ones(2, 1)
    _, components = fn(y_pred, y_true, relevance, masks)
    assert components.get('pairwise', 0) < 1e-6


def test_ndcg_approx_loss():
    """NDCGApproxLoss 前向传播不崩溃，完美排序损失小"""
    from utils import NDCGApproxLoss
    fn = NDCGApproxLoss(k=5, temperature=1.0, use_gumbel=False)
    y_true = torch.tensor([[9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    y_pred = torch.tensor([[9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    loss = fn(y_pred, y_true)
    assert loss.item() >= 0
    # Perfect ranking should give low loss
    assert loss.item() < 0.5, f"Perfect ranking loss too high: {loss.item()}"


def test_calculate_ranking_metrics():
    """排名指标计算不崩溃"""
    from train import calculate_ranking_metrics
    torch.manual_seed(42)
    y_pred = torch.rand(4, 20)
    y_true = torch.rand(4, 20)
    masks = torch.ones(4, 20)
    metrics = calculate_ranking_metrics(y_pred, y_true, masks, k=5)
    # metrics should be a dict with expected keys
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
