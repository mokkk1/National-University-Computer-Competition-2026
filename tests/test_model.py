"""
test_model.py — 模型前向传播单元测试
测试 StockTransformer 和 LightweightStockRanker 的输入输出形状、配置切换。
"""
import numpy as np
import torch
import pytest


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

# input_dim = 197 (158+39 base) + 7 (market breadth) = 204
# (no macro, no fundamental, no momentum, no industry to keep dim stable)
INPUT_DIM = 204


def _make_config(market=True):
    return {
        'sequence_length': 30, 'd_model': 64, 'nhead': 2, 'num_layers': 1,
        'dim_feedforward': 128, 'dropout': 0.1, 'feature_num': '158+39',
        'use_tcn': False, 'use_feature_interaction': False,
        'use_market_aggregation': market,
        'market_dim': 32, 'market_pool_heads': 2,
        'use_calendar_features': False,
        'use_fundamentals': False, 'use_momentum_features': False,
        'use_macro_features': False, 'use_industry_embedding': False,
    }


# ═══════════════════════════════════════════════════════════════
# Tests — StockTransformer
# ═══════════════════════════════════════════════════════════════

def test_stock_transformer_forward():
    """StockTransformer 前向传播形状正确"""
    from model import StockTransformer

    model = StockTransformer(input_dim=INPUT_DIM, config=_make_config(True), num_stocks=50)
    x = torch.randn(2, 50, 30, INPUT_DIM)

    model.eval()
    with torch.no_grad():
        scores, aux = model(x, return_aux=True)

    assert scores.shape == (2, 50)
    assert 'return_abs' in aux
    assert 'market_logits' in aux
    assert aux['market_logits'].shape[0] == 2  # [B] or [B, 1]


def test_stock_transformer_no_market():
    """关闭市场聚合时 market_logits 应为 None"""
    from model import StockTransformer

    model = StockTransformer(input_dim=INPUT_DIM, config=_make_config(False), num_stocks=50)
    x = torch.randn(2, 50, 30, INPUT_DIM)

    model.eval()
    with torch.no_grad():
        scores, aux = model(x, return_aux=True)

    # market_logits not in aux when market_aggregation is off
    assert 'market_logits' not in aux or aux['market_logits'] is None


def test_stock_transformer_param_count():
    """完整配置参数统计：~2.5M"""
    from model import StockTransformer

    full_cfg = {
        'sequence_length': 60, 'd_model': 256, 'nhead': 4, 'num_layers': 3,
        'dim_feedforward': 512, 'feature_num': '158+39+fundamental+momentum',
        'use_tcn': True, 'tcn_kernel_sizes': [3, 5, 7], 'tcn_dropout': 0.1,
        'use_feature_interaction': True, 'fi_rank': 64,
        'use_market_aggregation': True, 'market_dim': 64, 'market_pool_heads': 4,
        'use_calendar_features': False, 'dropout': 0.15,
        'use_fundamentals': True, 'use_momentum_features': True,
        'momentum_features': ['ret5', 'ret20', 'vol20', 'sharpe5'],
        'use_macro_features': True, 'use_industry_embedding': True,
        'num_industries': 31, 'industry_emb_dim': 16,
    }

    idim = 197 + 50 + 4 + 18 + 7  # base + fundamental + momentum + macro + breadth
    model = StockTransformer(input_dim=idim, config=full_cfg, num_stocks=300)
    total = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < total < 3_500_000, f"Unexpected param count: {total:,}"


# ═══════════════════════════════════════════════════════════════
# Tests — LightweightStockRanker
# ═══════════════════════════════════════════════════════════════

def test_lightweight_forward():
    """LightweightStockRanker 前向传播形状正确"""
    from model import LightweightStockRanker

    lw_cfg = {
        'sequence_length': 30, 'd_model': 64, 'gru_hidden': 32,
        'dropout': 0.2, 'feature_num': '158+39',
        'use_fundamentals': False, 'use_momentum_features': False,
        'use_macro_features': False, 'use_industry_embedding': False,
    }

    model = LightweightStockRanker(input_dim=INPUT_DIM, config=lw_cfg, num_stocks=50)
    x = torch.randn(2, 50, 30, INPUT_DIM)

    model.eval()
    with torch.no_grad():
        scores, aux = model(x, return_aux=True)

    assert scores.shape == (2, 50)
    assert 'return_abs' in aux


def test_lightweight_param_count():
    """轻量模型参数 < 500K"""
    from model import LightweightStockRanker

    lw_cfg = {
        'sequence_length': 60, 'd_model': 128, 'gru_hidden': 48,
        'feature_num': '158+39+fundamental+momentum', 'dropout': 0.2,
        'use_fundamentals': True, 'use_momentum_features': True,
        'momentum_features': ['ret5', 'ret20', 'vol20', 'sharpe5'],
        'use_macro_features': True, 'use_industry_embedding': True,
        'num_industries': 31, 'industry_emb_dim': 8,
    }

    idim = 197 + 50 + 4 + 18 + 7
    model = LightweightStockRanker(input_dim=idim, config=lw_cfg, num_stocks=300)
    total = sum(p.numel() for p in model.parameters())
    assert total < 500_000, f"Lightweight too large: {total:,} params"


def test_both_models_gradient_flow():
    """两个模型都能产生有效梯度"""
    from model import StockTransformer, LightweightStockRanker

    for ModelClass in [StockTransformer, LightweightStockRanker]:
        model = ModelClass(input_dim=INPUT_DIM, config=_make_config(False), num_stocks=20)
        x = torch.randn(1, 20, 30, INPUT_DIM, requires_grad=True)
        scores, aux = model(x, return_aux=True)
        loss = scores.mean() + aux['return_abs'].mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_strict_loading():
    """strict=False 加载兼容性"""
    from model import StockTransformer

    model = StockTransformer(input_dim=INPUT_DIM, config=_make_config(False), num_stocks=50)
    state = model.state_dict()
    state['_extra_old_key'] = torch.zeros(1)
    model.load_state_dict(state, strict=False)
    assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
