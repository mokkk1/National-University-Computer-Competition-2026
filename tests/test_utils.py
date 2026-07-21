"""
test_utils.py — 工具函数单元测试
测试市场宽度特征、懒加载数据集、后处理函数。
"""
import numpy as np
import pandas as pd
import torch
import pytest


# ═══════════════════════════════════════════════════════════════
# Test Data
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def small_stock_df():
    """构造小规模 stock_df。涨跌幅为百分比值（如 2.0=+2%）。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2025-01-06', periods=30, freq='B')
    stocks = [f'{i:06d}' for i in range(50)]
    rows = []
    for date in dates:
        for s in stocks:
            rows.append({
                '日期': date,
                '股票代码': s,
                'instrument': int(s),
                '开盘': rng.uniform(8, 25),
                '收盘': rng.uniform(8, 25),
                '最高': rng.uniform(8, 25),
                '最低': rng.uniform(8, 25),
                '成交量': rng.uniform(1e6, 1e8),
                '成交额': rng.uniform(1e7, 1e9),
                '涨跌幅': rng.uniform(-5.0, 5.0),
                '振幅': rng.uniform(1, 5),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Tests — Market Breadth
# ═══════════════════════════════════════════════════════════════

def test_compute_market_breadth_features(small_stock_df):
    """市场宽度特征计算：列存在 + 范围合理"""
    from utils import compute_market_breadth_features

    breadth = compute_market_breadth_features(small_stock_df)

    expected_cols = [
        'market_advance_ratio', 'market_return_dispersion',
        'market_return_skew', 'market_volume_change',
        'market_up_amount_ratio', 'market_amplitude_mean',
        'market_return_mean',
    ]
    for col in expected_cols:
        assert col in breadth.columns, f"Missing column: {col}"

    assert len(breadth) == small_stock_df['日期'].nunique()
    assert (breadth['market_advance_ratio'] >= 0).all()
    assert (breadth['market_advance_ratio'] <= 1).all()
    assert (breadth['market_return_dispersion'] >= 0).all()


def test_merge_market_breadth(small_stock_df):
    """合并市场宽度到个股，行数不变"""
    from utils import compute_market_breadth_features, merge_market_breadth

    breadth = compute_market_breadth_features(small_stock_df)
    merged = merge_market_breadth(small_stock_df, breadth)

    for col in breadth.columns:
        if col != '日期':
            assert col in merged.columns
    assert len(merged) == len(small_stock_df)


def test_advance_ratio_semantics():
    """涨跌比语义：全涨=1.0，全跌=0.0"""
    from utils import compute_market_breadth_features

    rows = []
    for date_str, ret in [('2025-01-06', 2.0), ('2025-01-07', -2.0)]:
        d = pd.Timestamp(date_str)
        for i in range(10):
            rows.append({
                '日期': d, '股票代码': f'{i:06d}', 'instrument': i,
                '开盘': 10.0, '收盘': 10.0, '最高': 10.0, '最低': 10.0,
                '成交量': 1e7, '成交额': 1e8, '振幅': 2.0,
                '涨跌幅': ret,
            })
    df = pd.DataFrame(rows)
    breadth = compute_market_breadth_features(df)

    assert breadth[breadth['日期'] == '2025-01-06']['market_advance_ratio'].iloc[0] == 1.0
    assert breadth[breadth['日期'] == '2025-01-07']['market_advance_ratio'].iloc[0] == 0.0


# ═══════════════════════════════════════════════════════════════
# Tests — Post-processing
# ═══════════════════════════════════════════════════════════════

def test_select_top_stocks_with_gate():
    """收益门控：只选预测正收益的股票"""
    from utils import select_top_stocks_with_gate

    rng = np.random.default_rng(42)
    n = 50
    scores = rng.uniform(0, 1, n)
    predicted_returns = np.concatenate([rng.uniform(0.01, 0.10, 10),
                                        rng.uniform(-0.05, -0.01, n - 10)])
    volatilities = rng.uniform(0.01, 0.08, n)

    indices, weights = select_top_stocks_with_gate(
        scores, predicted_returns, volatilities,
        top_k=5, candidate_k=20,
        min_return_threshold=0.0,
    )

    assert len(indices) == 5
    for idx in indices:
        assert predicted_returns[idx] > 0


def test_select_top_stocks_no_pos_return():
    """全负收益时回退到等权 Top-K"""
    from utils import select_top_stocks_with_gate

    rng = np.random.default_rng(42)
    n = 20
    scores = rng.uniform(0, 1, n)
    predicted_returns = np.full(n, -0.02)

    indices, weights = select_top_stocks_with_gate(
        scores, predicted_returns, None,
        top_k=3, min_return_threshold=0.0,
        fallback='equal',
    )
    assert len(indices) == 3


def test_optimize_weights():
    """权重优化：返回 (indices, weights) 两个数组，weights 和为1"""
    from utils import optimize_weights

    rng = np.random.default_rng(42)
    n = 5
    scores = rng.uniform(0.5, 1.0, n)
    vols = rng.uniform(0.01, 0.05, n)

    indices, weights = optimize_weights(scores, vols, top_k=5, candidate_k=5)
    assert len(indices) == 5
    assert len(weights) == 5
    assert abs(weights.sum() - 1.0) < 1e-6
    for w in weights:
        assert 0 <= w <= 1


# ═══════════════════════════════════════════════════════════════
# Tests — LazyRankingDataset
# ═══════════════════════════════════════════════════════════════

def test_lazy_dataset_basic():
    """懒加载数据集基本功能"""
    from utils import LazyRankingDataset, build_lazy_ranking_dataset

    rng = np.random.default_rng(42)
    n_stocks, n_dates, seq_len = 10, 30, 10
    feature_names = ['feat_a', 'feat_b', 'feat_c']

    dates = pd.date_range('2025-01-01', periods=n_dates, freq='B')

    rows = []
    for date in dates:
        for s in range(n_stocks):
            rows.append({
                '日期': date,
                'instrument': s,
                'label': rng.uniform(0, 1),
                'label_abs': rng.uniform(-0.05, 0.1),
                'market_label': float(rng.integers(0, 2)),
                **{f: rng.normal(0, 1) for f in feature_names},
            })
    df = pd.DataFrame(rows)

    dataset = build_lazy_ranking_dataset(
        df, feature_names, seq_len,
        min_window_end_date=dates[seq_len],
        max_future_span_days=15,
    )

    assert dataset is not None
    assert len(dataset) > 0

    sample = dataset[0]
    assert 'sequences' in sample
    assert 'targets' in sample
    assert 'relevance' in sample
    assert sample['sequences'].shape[0] == n_stocks  # num_stocks per day
    assert sample['sequences'].shape[1] == seq_len     # sequence_length
    assert sample['sequences'].shape[2] == len(feature_names)


def test_lazy_dataset_collate():
    """懒加载数据集与 collate_fn 兼容"""
    from utils import LazyRankingDataset, build_lazy_ranking_dataset
    from train import collate_fn

    rng = np.random.default_rng(42)
    n_stocks, n_dates, seq_len = 10, 25, 10
    feature_names = ['f1', 'f2', 'f3']
    dates = pd.date_range('2025-01-01', periods=n_dates, freq='B')

    rows = []
    for date in dates:
        for s in range(n_stocks):
            rows.append({
                '日期': date,
                'instrument': s,
                'label': rng.uniform(0, 1),
                'label_abs': rng.uniform(-0.05, 0.1),
                'market_label': float(rng.integers(0, 2)),
                **{f: rng.normal(0, 1) for f in feature_names},
            })
    df = pd.DataFrame(rows)

    dataset = build_lazy_ranking_dataset(
        df, feature_names, seq_len,
        min_window_end_date=dates[seq_len],
        max_future_span_days=15,
    )

    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    collated = collate_fn(batch)

    assert 'sequences' in collated
    assert 'targets' in collated
    assert collated['sequences'].shape[1] == n_stocks
    assert collated['sequences'].shape[2] == seq_len
    assert collated['sequences'].shape[3] == len(feature_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
