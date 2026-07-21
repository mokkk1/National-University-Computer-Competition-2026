"""
fundamental.py — 基本面特征工程模块

提供:
  1. 基本面数据加载 (数据源: data/fundamentals.csv)
  2. 基本面特征工程 (估值、盈利、成长、资金流向)
  3. 组合特征工程函数

与 utils.py 配合使用，不修改 utils.py 核心代码。
"""

import pandas as pd
import numpy as np
import os
import sys

_fundamental_cache = None


def load_fundamentals(force_reload=False):
    """加载基本面缓存数据（带内存缓存避免重复IO）"""
    global _fundamental_cache
    if _fundamental_cache is not None and not force_reload:
        return _fundamental_cache
    # 查找 fundamental.csv
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fundamentals.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'fundamentals.csv'),
    ]
    for cache_path in candidates:
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['日期'])
            _fundamental_cache = df
            return df
    return None


def engineer_fundamental_features(df, fundamentals_df=None):
    """
    为DataFrame添加10维基本面特征。

    Args:
        df: 包含 ['日期', '股票代码'] + 量价特征的DataFrame
        fundamentals_df: 可选，预先加载的基本面数据

    Returns:
        添加了基本面特征的DataFrame (原始列 + 基本面列)
    """
    if fundamentals_df is None:
        fundamentals_df = load_fundamentals()

    df = df.copy()

    # 基本面核心特征列表
    fund_feature_cols = [
        'pe_rank', 'pb_rank', 'ps_rank',         # 估值分位数
        'roe', 'roa', 'gross_margin',             # 盈利能力
        'revenue_yoy', 'profit_yoy',              # 成长性
        'north_holding_pct',                      # 北向持股
        'fund_flow_5d',                           # 资金流向
        'industry_id',                            # 行业编码
    ]

    if fundamentals_df is None or len(fundamentals_df) == 0:
        for col in fund_feature_cols:
            df[col] = 0.0
        return df

    fundamentals_df = fundamentals_df.copy()
    fundamentals_df['日期'] = pd.to_datetime(fundamentals_df['日期'])
    df['日期'] = pd.to_datetime(df['日期'])

    # 统一股票代码类型
    try:
        fundamentals_df['股票代码'] = fundamentals_df['股票代码'].astype(df['股票代码'].dtype)
    except (KeyError, ValueError):
        return _fill_zero_fundamentals(df, fund_feature_cols)

    # 合并
    merge_cols = ['日期', '股票代码']
    fund_cols = [c for c in fundamentals_df.columns
                 if c not in merge_cols and c not in df.columns]

    if not fund_cols:
        return _fill_zero_fundamentals(df, fund_feature_cols)

    df = pd.merge(df, fundamentals_df[merge_cols + fund_cols],
                  on=merge_cols, how='left')

    # 按股票前向填充
    for col in fund_cols:
        if col in df.columns:
            df[col] = df.groupby('股票代码')[col].ffill()
            df[col] = df[col].fillna(0)

    # ─── 1. 估值分位数（截面排名） ───
    for col in ['pe', 'pb', 'ps']:
        if col in df.columns:
            raw = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
            df[f'{col}_rank'] = df.groupby('日期')[col].rank(pct=True)
            df[f'{col}_rank'] = df[f'{col}_rank'].fillna(0.5)
        else:
            df[f'{col}_rank'] = 0.0

    # ─── 2. ROE/ROA/毛利率 ───
    for col in ['roe', 'roa', 'gross_margin']:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0)
        else:
            df[col] = 0.0

    # ─── 3. 营收/利润增速 ───
    for col in ['revenue_yoy', 'profit_yoy']:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0)
        else:
            df[col] = 0.0

    # ─── 4. 北向持股 ───
    if '持股比例' in df.columns:
        df['north_holding_pct'] = df['持股比例'].astype(float).fillna(0)
    else:
        df['north_holding_pct'] = 0.0

    # ─── 5. 资金流向5日累计 ───
    if '主力净流入-净额' in df.columns:
        df['fund_flow_5d'] = df.groupby('股票代码')['主力净流入-净额'].transform(
            lambda x: x.rolling(5, min_periods=1).sum().fillna(0)
        )
    else:
        df['fund_flow_5d'] = 0.0

    # ─── 6. 行业编码 ───
    if '行业' in df.columns:
        df['industry_label'] = df['行业'].apply(
            lambda x: x.split(',')[0] if isinstance(x, str) and x else 'unknown'
        )
        df['industry_id'] = df.groupby('industry_label').ngroup()
        df.drop(columns=['industry_label'], inplace=True)
    else:
        df['industry_id'] = 0

    # 清理
    for col in fund_feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        else:
            df[col] = 0.0

    return df


def _fill_zero_fundamentals(df, feature_cols):
    """无基本面数据时填充零"""
    for col in feature_cols:
        df[col] = 0.0
    return df


def engineer_features_158plus39_fundamental(df, fundamentals_df=None):
    """
    完整特征工程: 158 Alpha + 39 技术指标 + 基本面特征

    这是推荐的特征工程入口函数，兼容原有接口。
    """
    # 延迟导入避免循环依赖
    from utils import engineer_features_158plus39

    # 1. 基础量价特征
    df_with_tech = engineer_features_158plus39(df)

    # 2. 添加基本面特征
    df_final = engineer_fundamental_features(df_with_tech, fundamentals_df)

    # 3. 去重
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # 4. 全局清理
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)

    return df_final


def engineer_momentum_features(df):
    """
    添加简单动量特征: ret5, ret20, vol20, sharpe5

    这些特征是朴素策略中表现最好的信号（单因子7.37%），
    直接加入特征让模型学习如何与复杂特征组合。
    """
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 收益率
    df['ret1'] = df.groupby('股票代码')['收盘'].pct_change(1)
    df['ret5'] = df.groupby('股票代码')['收盘'].pct_change(5)
    df['ret20'] = df.groupby('股票代码')['收盘'].pct_change(20)

    # 波动率
    df['vol5'] = df.groupby('股票代码')['ret1'].transform(
        lambda x: x.rolling(5).std())
    df['vol20'] = df.groupby('股票代码')['ret1'].transform(
        lambda x: x.rolling(20).std())

    # 夏普比率
    df['sharpe5'] = df['ret5'] / (df['vol20'] + 0.001)
    df['sharpe20'] = df['ret20'] / (df['vol20'] + 0.001)

    # 成交量变化
    df['vol_chg'] = df.groupby('股票代码')['成交量'].pct_change(5)

    # 换手率变化
    if '换手率' in df.columns:
        df['turn_chg'] = df.groupby('股票代码')['换手率'].pct_change(5)

    # 近期最大收益/最大回撤
    df['max_ret5'] = df.groupby('股票代码')['ret1'].transform(
        lambda x: x.rolling(5).max())
    df['min_ret5'] = df.groupby('股票代码')['ret1'].transform(
        lambda x: x.rolling(5).min())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def engineer_features_all(df, fundamentals_df=None):
    """
    完整特征工程: 158 Alpha + 39 技术指标 + 基本面 + 动量
    """
    from utils import engineer_features_158plus39

    # 1. 基础量价特征
    df_tech = engineer_features_158plus39(df)

    # 2. 基本面
    df_fund = engineer_fundamental_features(df_tech, fundamentals_df)

    # 3. 动量特征
    df_final = engineer_momentum_features(df_fund)

    # 4. 去重清理
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)

    return df_final


# ─── 特征列映射 ──────────────────────────────────────
FUNDAMENTAL_FEATURE_COLS = [
    'pe_rank', 'pb_rank', 'ps_rank',
    'roe', 'roa', 'gross_margin',
    'revenue_yoy', 'profit_yoy',
    'north_holding_pct', 'fund_flow_5d',
    'industry_id',
]

MOMENTUM_FEATURE_COLS = [
    'ret1', 'ret5', 'ret20',
    'vol5', 'vol20',
    'sharpe5', 'sharpe20',
    'vol_chg', 'turn_chg',
    'max_ret5', 'min_ret5',
]
