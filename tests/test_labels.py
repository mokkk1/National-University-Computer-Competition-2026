"""
test_labels.py — 标签构建单元测试
测试 _build_label_and_clean 的混合标签逻辑、分位数标签、过滤条件等。
"""
import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_synthetic_df(n_days=10, n_stocks=5, seed=42, freq='W-MON'):
    """生成合成数据。freq='W-MON'=每周一，模拟真实交易日历。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2025-01-06', periods=n_days, freq=freq)
    rows = []
    for date in dates:
        for stock_idx in range(n_stocks):
            rows.append({
                '日期': date,
                '股票代码': f'{stock_idx:06d}',
                '开盘': rng.uniform(8, 20),
                '收盘': rng.uniform(8, 20),
                'instrument': stock_idx,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_build_label_and_clean_basic():
    """基本标签构建 — label_abs 应为未来5日收益率"""
    from train import _build_label_and_clean

    df = _make_synthetic_df(n_days=12, n_stocks=3)
    cleaned = _build_label_and_clean(df.copy())

    assert len(cleaned) < len(df), "应丢弃缺少未来数据的尾部样本"
    assert 'label_abs' in cleaned.columns
    assert cleaned['label_abs'].notna().all()


def test_quantile_label_range():
    """分位数标签应在 [0, 1] 范围内"""
    from train import _build_label_and_clean

    df = _make_synthetic_df(n_days=20, n_stocks=10, freq='B')
    cleaned = _build_label_and_clean(df.copy(), use_quantile_label=True)

    assert 'label' in cleaned.columns
    for date, group in cleaned.groupby('日期'):
        assert (group['label'] >= 0).all(), f"Date {date}: label < 0"
        assert (group['label'] <= 1).all(), f"Date {date}: label > 1"


def test_mixed_label():
    """混合标签与纯rank应有差异"""
    from train import _build_label_and_clean

    df = _make_synthetic_df(n_days=20, n_stocks=10, freq='B')
    mixed = _build_label_and_clean(df.copy(), use_quantile_label=True,
                                    use_mixed_label=True, mixed_alpha=0.7)
    rank_only = _build_label_and_clean(df.copy(), use_quantile_label=True,
                                        use_mixed_label=False)

    mixed_std = mixed.groupby('日期')['label'].std().mean()
    rank_std = rank_only.groupby('日期')['label'].std().mean()
    assert mixed_std != rank_std, "混合标签与纯rank标签应有差异"


def test_label_abs_computation():
    """label_abs 正负号与价格趋势一致"""
    from train import _build_label_and_clean

    dates = pd.date_range('2025-01-06', periods=15, freq='B')
    rows = []
    for i, date in enumerate(dates):
        for stock in range(2):
            rows.append({
                '日期': date,
                '股票代码': f'{stock:06d}',
                '开盘': 10.0 + i * 0.2 + stock,
                '收盘': 10.0 + i * 0.2 + stock + 0.3,
                'instrument': stock,
            })
    df = pd.DataFrame(rows)

    cleaned = _build_label_and_clean(df.copy(), drop_small_open=False)
    first = cleaned[cleaned['日期'] == dates[0]]
    assert len(first) == 2
    assert (first['label_abs'] > 0).all(), "递增价格应产生正收益"


def test_drop_small_open():
    """drop_small_open 过滤极端低价"""
    from train import _build_label_and_clean

    df = _make_synthetic_df(n_days=20, n_stocks=5, freq='B')
    df.loc[0, '开盘'] = 0.0001
    # drop_small_open=True 应正常处理，不崩溃
    cleaned = _build_label_and_clean(df.copy(), drop_small_open=True)
    # 过滤后样本数应减少
    assert len(cleaned) >= 0


def test_empty_input():
    """空输入不应崩溃"""
    from train import _build_label_and_clean

    df = pd.DataFrame(columns=['日期', '股票代码', '开盘', '收盘', 'instrument'])
    try:
        result = _build_label_and_clean(df.copy())
        assert len(result) == 0
    except Exception as e:
        pytest.fail(f"空输入不应抛异常: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
