"""
test_market_gate.py — 市场门控后处理单元测试
测试 market_signal computation, quarter_aware strategy, defensive scoring.
"""
import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_market_df(dates_returns):
    """构造有明确涨跌幅的 market DataFrame。
    dates_returns: [(date_str, pct_change), ...]  其中 pct_change 是百分比值（如 2.0 = +2%）
    """
    stocks = ['000001', '000002', '600036']
    rows = []
    for date_str, ret_pct in dates_returns:
        d = pd.Timestamp(date_str)
        for s in stocks:
            rows.append({
                '日期': d,
                '股票代码': s,
                '开盘': 10.0,
                '收盘': 10.0,
                '涨跌幅': ret_pct,
                '成交量': 5e7,
                '成交额': 5e8,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Tests — HS300 Return
# ═══════════════════════════════════════════════════════════════

def test_cumulative_hs300_return():
    """累计涨跌幅计算"""
    from market_gate import compute_hs300_return

    dates_rets = [(f'2025-01-{d:02d}', 2.0) for d in range(6, 21) if d not in (11, 12, 18, 19)]
    df = _make_market_df(dates_rets)
    result = compute_hs300_return(df, pd.Timestamp('2025-01-20'), lookback=10)
    assert result > 0
    assert result < 0.5


def test_hs300_return_lookback_limit():
    """lookback 超出数据范围时应返回有限值"""
    from market_gate import compute_hs300_return

    dates_rets = [(f'2025-01-{d:02d}', 1.0) for d in [6, 7, 8]]
    df = _make_market_df(dates_rets)
    result = compute_hs300_return(df, pd.Timestamp('2025-01-08'), lookback=10)
    assert isinstance(result, float)
    assert abs(result) < 1.0


# ═══════════════════════════════════════════════════════════════
# Tests — Market Signal
# ═══════════════════════════════════════════════════════════════

def test_market_signal_structure():
    """市场信号返回结构正确"""
    from market_gate import compute_market_signal

    dates_rets = [(f'2025-01-{d:02d}', r) for d, r in
                  zip(range(6, 17), [2.0, -1.0, 1.5, -0.5, 2.5, 1.0,
                                     -2.0, 0.8, 1.2, -1.5, 1.8])]
    df = _make_market_df(dates_rets)
    signal = compute_market_signal(df, pd.Timestamp('2025-01-16'), method='hs300_return')

    for key in ['direction', 'signal', 'confidence', 'method', 'sub_signals']:
        assert key in signal, f"缺少字段: {key}"
    assert signal['direction'] in [-1, 0, 1]


def test_market_signal_bull():
    """明确上涨：direction=1, signal>0"""
    from market_gate import compute_market_signal

    dates_rets = [(f'2025-01-{d:02d}', 3.0) for d in range(6, 21)]
    df = _make_market_df(dates_rets)
    signal = compute_market_signal(df, pd.Timestamp('2025-01-20'), method='hs300_return')
    assert signal['direction'] == 1
    assert signal['signal'] > 0.01


def test_market_signal_bear():
    """明确下跌：direction=-1, signal<0"""
    from market_gate import compute_market_signal

    dates_rets = [(f'2025-01-{d:02d}', -3.0) for d in range(6, 21)]
    df = _make_market_df(dates_rets)
    signal = compute_market_signal(df, pd.Timestamp('2025-01-20'), method='hs300_return')
    assert signal['direction'] == -1
    assert signal['signal'] < -0.01


def test_market_signal_flat():
    """平稳市场：direction=0"""
    from market_gate import compute_market_signal

    dates_rets = [(f'2025-01-{d:02d}', 0.08) for d in range(6, 21)]  # ~0.8% cum
    df = _make_market_df(dates_rets)
    signal = compute_market_signal(df, pd.Timestamp('2025-01-20'), method='hs300_return')
    assert signal['direction'] == 0
    assert abs(signal['signal']) < 0.02


# ═══════════════════════════════════════════════════════════════
# Tests — MarketGate
# ═══════════════════════════════════════════════════════════════

def test_market_gate_days_to_qe():
    """_compute_days_to_qe 返回值正确"""
    from market_gate import MarketGate

    gate = MarketGate()
    assert gate._compute_days_to_qe(pd.Timestamp('2024-09-30')) == 0.0
    assert gate._compute_days_to_qe(pd.Timestamp('2024-12-31')) == 0.0

    days = gate._compute_days_to_qe(pd.Timestamp('2024-09-26'))
    assert 0 < days < 0.06

    days = gate._compute_days_to_qe(pd.Timestamp('2024-07-15'))
    assert days > 0.5


def test_adaptive_defensive_weight_quarter_end_bear():
    """季末+看跌: 防御权重应很高"""
    from market_gate import MarketGate

    gate = MarketGate(strategy='quarter_aware')
    w = gate._adaptive_defensive_weight(direction=-1, confidence=0.9, days_to_qe=0.03)
    assert w >= 0.85, f"季末看跌应有高防御权重, got {w}"


def test_adaptive_defensive_weight_normal_bull():
    """非季末+看涨: 防御权重应很低"""
    from market_gate import MarketGate

    gate = MarketGate(strategy='quarter_aware')
    w = gate._adaptive_defensive_weight(direction=1, confidence=0.8, days_to_qe=0.8)
    assert w < 0.2, f"普通看涨应有低防御权重, got {w}"


def test_adaptive_defensive_weight_neutral():
    """中性信号应有中等防御"""
    from market_gate import MarketGate

    gate = MarketGate(strategy='quarter_aware')
    w = gate._adaptive_defensive_weight(direction=0, confidence=0.5, days_to_qe=0.5)
    assert 0.2 < w < 0.5, f"中性信号应中等, got {w}"


def test_market_gate_invalid_strategy_fallback():
    """无效策略名不应崩溃"""
    from market_gate import MarketGate

    gate = MarketGate(strategy='nonexistent')
    assert gate.strategy == 'nonexistent'


def test_defensive_score_computation():
    """防御性评分计算"""
    from market_gate import compute_stock_defensive_score

    rng = np.random.default_rng(42)
    dates = pd.date_range('2025-01-06', periods=30, freq='B')
    rows = []
    for d in dates:
        for s in ['000001', '000002', '600036']:
            rows.append({
                '日期': d,
                '股票代码': s,
                '涨跌幅': rng.uniform(-3, 3),
                '成交额': rng.uniform(1e8, 1e9),
            })
    df = pd.DataFrame(rows)

    scores = compute_stock_defensive_score(['000001', '000002', '600036'], df, dates[-1])
    assert len(scores) == 3
    for s in scores:
        assert 0 <= s <= 1
    assert len(set(np.round(scores, 4))) > 1, "分数不应全相等"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
