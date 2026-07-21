"""
fetch_macro.py — 获取宏观指标数据

数据来源（均为 AKShare 免费接口）：
  日频：
  - 10年期国债收益率: bond_china_yield
  - Shibor隔夜拆借利率: rate_interbank
  - 北向资金净流入: stock_hsgt_north_net_flow_in_em
  - 融资余额(沪市): stock_margin_sse
  - 美元/人民币中间价: currency_boc_sina
  - 7天逆回购利率: macro_china_lpr (用 LPR 1Y 近似)

  月频（前向填充到日频）：
  - M1/M2 同比: macro_china_money_supply
  - CPI 同比: macro_china_cpi
  - PPI 同比: macro_china_ppi
  - PMI: macro_china_pmi
  - 社融增量: macro_china_shrzgm

用法：
  python fetch_macro.py                    # 首次下载
  python fetch_macro.py --force            # 强制刷新
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_DIR, 'data', 'macro_features.csv')
META_FILE = os.path.join(PROJECT_DIR, 'data', 'macro_meta.json')


def safe_fetch(func, name, **kwargs):
    """安全调用 akshare，失败返回 None"""
    try:
        print(f"  [{name}] 获取中...")
        result = func(**kwargs)
        if result is not None and isinstance(result, pd.DataFrame) and len(result) > 0:
            print(f"  [{name}] ✅ {len(result)} 行")
            return result
        else:
            print(f"  [{name}] ⚠️ 空数据")
            return None
    except Exception as e:
        print(f"  [{name}] ❌ {e}")
        return None


def fetch_bond_yield():
    """中美10年期国债收益率 (日频) — 用 bond_zh_us_rate"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate()
        if df is not None and len(df) > 0:
            df = df[['日期', '中国国债收益率10年']].copy()
            df.columns = ['日期', 'bond10y_yield']
            df['日期'] = pd.to_datetime(df['日期'])
            df['bond10y_yield'] = pd.to_numeric(df['bond10y_yield'], errors='coerce')
            df = df.dropna().sort_values('日期')
            return df
    except Exception:
        pass
    return None


def fetch_north_flow():
    """北向资金净买入 (日频) — 用 stock_hsgt_hist_em"""
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em(symbol='北向资金')
        if df is not None and len(df) > 0:
            df = df[['日期', '当日成交净买额']].copy()
            df.columns = ['日期', 'north_flow_in']
            df['日期'] = pd.to_datetime(df['日期'])
            df['north_flow_in'] = pd.to_numeric(df['north_flow_in'], errors='coerce')
            df = df.sort_values('日期')
            df['north_flow_5d'] = df['north_flow_in'].rolling(5, min_periods=1).sum()
            df['north_flow_20d'] = df['north_flow_in'].rolling(20, min_periods=1).sum()
            return df
    except Exception:
        pass
    return None


def fetch_margin_balance():
    """融资余额 (日频)"""
    try:
        import akshare as ak
        dfs = []
        for market, name in [("sh", "沪市"), ("sz", "深市")]:
            try:
                df = ak.stock_margin_sse(start_date='20100101') if market == 'sh' else \
                     ak.stock_margin_szse(start_date='20100101') if market == 'sz' else None
                if df is not None and len(df) > 0:
                    date_col = df.columns[0]
                    balance_col = [c for c in df.columns if '余额' in c or '融资' in c]
                    if balance_col:
                        col = balance_col[0]
                        df = df[[date_col, col]].copy()
                        df.columns = ['日期', f'margin_{market}']
                        df['日期'] = pd.to_datetime(df['日期'])
                        dfs.append(df)
            except Exception:
                pass
        if dfs:
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on='日期', how='outer')
            # 融资余额变化率
            if 'margin_sh' in result.columns:
                result['margin_change'] = result['margin_sh'].pct_change(5) * 100
            return result.sort_values('日期')
    except Exception:
        pass
    return None


def fetch_exchange_rate():
    """美元/人民币中间价 (日频)"""
    try:
        import akshare as ak
        df = ak.currency_boc_sina(symbol="美元")
        if df is not None and len(df) > 0:
            if '日期' in df.columns and '中间价' in df.columns:
                df = df[['日期', '中间价']].copy()
            elif len(df.columns) >= 2:
                df = df.iloc[:, :2].copy()
            else:
                return None
            df.columns = ['日期', 'usdcny']
            df['日期'] = pd.to_datetime(df['日期'])
            df['usdcny'] = pd.to_numeric(df['usdcny'], errors='coerce')
            # 汇率变化率
            df['usdcny_chg_5d'] = df['usdcny'].pct_change(5) * 100
            df['usdcny_chg_20d'] = df['usdcny'].pct_change(20) * 100
            return df
    except Exception:
        pass
    return None


def fetch_lpr():
    """LPR 利率 (月频 → 前向填充)"""
    try:
        import akshare as ak
        df = ak.macro_china_lpr()
        if df is not None and len(df) > 0:
            date_col = df.columns[0]
            # 1年期 LPR
            lpr_col = [c for c in df.columns if '1年' in c or '一年' in c or 'LPR' in c]
            if not lpr_col:
                lpr_col = [df.columns[1]] if len(df.columns) > 1 else [date_col]
            df = df[[date_col, lpr_col[0]]].copy()
            df.columns = ['日期', 'lpr_1y']
            df['日期'] = pd.to_datetime(df['日期'])
            df['lpr_1y'] = pd.to_numeric(df['lpr_1y'], errors='coerce')
            return df
    except Exception:
        pass
    return None


def fetch_shibor():
    """Shibor 隔夜拆借利率 (日频)"""
    try:
        import akshare as ak
        df = ak.rate_interbank(market='上海银行同业拆借市场', symbol='Shibor人民币', indicator='隔夜')
        if df is not None and len(df) > 0:
            df = df[['报告日', '利率']].copy()
            df.columns = ['日期', 'shibor_on']
            df['日期'] = pd.to_datetime(df['日期'])
            df['shibor_on'] = pd.to_numeric(df['shibor_on'], errors='coerce')
            return df.sort_values('日期')
    except Exception:
        pass
    return None


def fetch_cpi():
    """CPI 同比 (月频)"""
    try:
        import akshare as ak
        df = ak.macro_china_cpi_yearly()
        if df is not None and len(df) > 0:
            cpi = df[df['商品'] == '中国CPI年率报告'].copy()
            if len(cpi) > 0:
                cpi = cpi[['日期', '今值']].copy()
                cpi.columns = ['日期', 'cpi_yoy']
                cpi['日期'] = pd.to_datetime(cpi['日期'])
                cpi['cpi_yoy'] = pd.to_numeric(cpi['cpi_yoy'], errors='coerce')
                return cpi.sort_values('日期')
    except Exception:
        pass
    return None


def fetch_ppi():
    """PPI 同比 (月频)"""
    try:
        import akshare as ak
        df = ak.macro_china_ppi_yearly()
        if df is not None and len(df) > 0:
            ppi = df[df['商品'] == '中国PPI年率报告'].copy()
            if len(ppi) > 0:
                ppi = ppi[['日期', '今值']].copy()
                ppi.columns = ['日期', 'ppi_yoy']
                ppi['日期'] = pd.to_datetime(ppi['日期'])
                ppi['ppi_yoy'] = pd.to_numeric(ppi['ppi_yoy'], errors='coerce')
                return ppi.sort_values('日期')
    except Exception:
        pass
    return None


def fetch_pmi():
    """PMI (月频) — AKShare v2: 列名 '月份' 格式 '2026年06月份'"""
    try:
        import akshare as ak
        df = ak.macro_china_pmi()
        if df is not None and len(df) > 0:
            result = pd.DataFrame()
            result['日期'] = pd.to_datetime(
                df['月份'].str.replace('年', '-').str.replace('月份', '')
            )
            result['pmi_mfg'] = pd.to_numeric(df['制造业-指数'], errors='coerce')
            result['pmi_non_mfg'] = pd.to_numeric(df['非制造业-指数'], errors='coerce')
            return result.sort_values('日期')
    except Exception:
        pass
    return None


def fetch_money_supply():
    """M1/M2 同比 (月频) — AKShare v2: 列名 '月份', 明确列名"""
    try:
        import akshare as ak
        df = ak.macro_china_money_supply()
        if df is not None and len(df) > 0:
            result = pd.DataFrame()
            result['日期'] = pd.to_datetime(
                df['月份'].str.replace('年', '-').str.replace('月份', '')
            )
            result['m1_yoy'] = pd.to_numeric(df['货币(M1)-同比增长'], errors='coerce')
            result['m2_yoy'] = pd.to_numeric(df['货币和准货币(M2)-同比增长'], errors='coerce')
            return result.sort_values('日期')
    except Exception:
        pass
    return None


def fetch_social_financing():
    """社融增量 (月频) — AKShare v2: 列名 '月份' 格式 '201501'"""
    try:
        import akshare as ak
        df = ak.macro_china_shrzgm()
        if df is not None and len(df) > 0:
            result = pd.DataFrame()
            result['日期'] = pd.to_datetime(df['月份'].astype(str), format='%Y%m')
            result['social_finance'] = pd.to_numeric(df['社会融资规模增量'], errors='coerce')
            return result.sort_values('日期')
    except Exception:
        pass
    return None


def build_macro_dataset(force=False):
    """构建完整的宏观特征数据集"""
    os.makedirs(os.path.join(PROJECT_DIR, 'data'), exist_ok=True)

    # 缓存检查（24小时有效）
    if not force and os.path.exists(OUTPUT_FILE) and os.path.exists(META_FILE):
        with open(META_FILE) as f:
            meta = json.load(f)
        if (datetime.now() - datetime.fromisoformat(meta['date'])).total_seconds() < 86400:
            print(f"缓存有效 ({meta['date']})，跳过下载。使用 --force 强制刷新。")
            return pd.read_csv(OUTPUT_FILE, parse_dates=['日期'])

    print("=" * 60)
    print("获取宏观指标数据...")
    print("=" * 60)

    all_data = {}

    # ─── 日频数据 ──────────────────────────────
    print("\n── 日频指标 ──")
    bond = safe_fetch(fetch_bond_yield, "10Y国债收益率")
    if bond is not None:
        all_data['bond'] = bond

    shibor = safe_fetch(fetch_shibor, "Shibor隔夜")
    if shibor is not None:
        all_data['shibor'] = shibor

    north = safe_fetch(fetch_north_flow, "北向资金")
    if north is not None:
        all_data['north'] = north

    margin = safe_fetch(fetch_margin_balance, "融资余额")
    if margin is not None:
        all_data['margin'] = margin

    fx = safe_fetch(fetch_exchange_rate, "美元/人民币")
    if fx is not None:
        all_data['fx'] = fx

    # ─── 月频数据 ──────────────────────────────
    print("\n── 月频指标 ──")
    lpr = safe_fetch(fetch_lpr, "LPR")
    if lpr is not None:
        all_data['lpr'] = lpr

    cpi = safe_fetch(fetch_cpi, "CPI同比")
    if cpi is not None:
        all_data['cpi'] = cpi

    ppi = safe_fetch(fetch_ppi, "PPI同比")
    if ppi is not None:
        all_data['ppi'] = ppi

    pmi = safe_fetch(fetch_pmi, "PMI")
    if pmi is not None:
        all_data['pmi'] = pmi

    m2 = safe_fetch(fetch_money_supply, "M1/M2货币供应")
    if m2 is not None:
        all_data['m2'] = m2

    sf = safe_fetch(fetch_social_financing, "社融增量")
    if sf is not None:
        all_data['sf'] = sf

    # ─── 合并 ─────────────────────────────────
    print("\n── 合并数据 ──")
    # 创建完整的日期索引（2010 起，与扩展后的行情数据对齐；
    # LPR(2019+)/北向(2014+)等晚出现的指标，早期按惯例 ffill 后填 0）
    date_range = pd.date_range(start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), freq='B')
    merged = pd.DataFrame({'日期': date_range})

    for name, df in all_data.items():
        if df is None or len(df) == 0:
            continue
        merged = pd.merge(merged, df, on='日期', how='left')

    merged = merged.sort_values('日期').reset_index(drop=True)

    # 前向填充月频数据
    month_fill_cols = []
    for c in merged.columns:
        if c != '日期' and merged[c].notna().sum() < len(merged) * 0.1:
            month_fill_cols.append(c)

    print(f"  月频列（前向填充）: {month_fill_cols}")
    for col in month_fill_cols:
        merged[col] = merged[col].ffill()

    # 日频缺失值用最近值填充，最后填0
    merged = merged.ffill().fillna(0)

    # 保存
    merged.to_csv(OUTPUT_FILE, index=False)
    with open(META_FILE, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'rows': len(merged),
            'columns': list(merged.columns),
            'sources': list(all_data.keys()),
        }, f)

    print(f"\n✅ 宏观数据已保存: {OUTPUT_FILE}")
    print(f"   行数: {len(merged):,}")
    print(f"   列数: {len(merged.columns)}")
    print(f"   列名: {list(merged.columns)}")
    print(f"   日期范围: {merged['日期'].min().date()} ~ {merged['日期'].max().date()}")

    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='获取宏观指标')
    parser.add_argument('--force', action='store_true', help='强制刷新')
    args = parser.parse_args()
    build_macro_dataset(force=args.force)
