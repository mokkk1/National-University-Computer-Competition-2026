"""
fetch_fundamentals.py — 使用 baostock 快速获取基本面数据

数据来源 (baostock, 已安装):
  1. query_stock_industry() → 行业分类
  2. query_profit_data() → ROE, ROA, 毛利率, 净利率
  3. query_growth_data() → 营收/利润增速

输出: data/fundamentals.csv
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(DATA_DIR, 'data', 'fundamentals.csv')
os.makedirs(os.path.join(DATA_DIR, 'data'), exist_ok=True)


def load_stock_list():
    """加载沪深300成分股"""
    # 从 train.csv 获取
    train_path = os.path.join(DATA_DIR, 'train.csv')
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        codes = sorted(df['股票代码'].unique())
        return [str(c) for c in codes]
    # 从 stock_data 获取
    sd_path = os.path.join(DATA_DIR, 'stock_data.csv')
    if os.path.exists(sd_path):
        df = pd.read_csv(sd_path)
        return sorted(df['股票代码'].unique().astype(str))
    return []


def baostock_login():
    import baostock as bs
    lg = bs.login()
    if lg.error_code != '0':
        print(f'Baostock login error: {lg.error_msg}')
        return False
    return True


def baostock_logout():
    import baostock as bs
    bs.logout()


def _to_bs_code(code):
    """Convert numeric code to baostock format (sh.600000 or sz.000001)"""
    code = str(code).zfill(6)
    if code.startswith('6') or code.startswith('5'):
        return f'sh.{code}'
    else:
        return f'sz.{code}'


def fetch_industry_data(stock_codes):
    """获取行业分类"""
    import baostock as bs
    records = []
    for code in tqdm(stock_codes, desc="Industry"):
        try:
            bs_code = _to_bs_code(code)
            rs = bs.query_stock_industry(bs_code)
            if rs.error_code == '0':
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=rs.fields)
                    # 取最新的一条
                    latest = df.sort_values('updateDate').iloc[-1] if 'updateDate' in df.columns else df.iloc[0]
                    records.append({
                        '股票代码': code,
                        '行业': latest.get('industry', ''),
                        '行业分类': latest.get('industryClassify', ''),
                    })
        except Exception as e:
            pass
    return pd.DataFrame(records) if records else pd.DataFrame()


def fetch_profit_data(stock_codes):
    """获取盈利能力数据: ROE, ROA, 毛利率, 净利率"""
    import baostock as bs
    all_records = []
    for code in tqdm(stock_codes, desc="Profit"):
        try:
            bs_code = _to_bs_code(code)
            rs = bs.query_profit_data(bs_code, year=2024, quarter=4)
            if rs.error_code == '0':
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=rs.fields)
                    for _, row in df.iterrows():
                        all_records.append({
                            '股票代码': code,
                            '日期': f"{row.get('statDate', '')}",
                            'roe': float(row.get('roeAvg', 0) or 0),
                            'roa': float(row.get('roa', 0) or 0),
                            'gross_margin': float(row.get('grossProfitRatio', 0) or 0),
                            'net_margin': float(row.get('netProfitMargin', 0) or 0),
                        })
        except Exception:
            pass

        # 也获取2023和2025年的数据
        for yr in [2023, 2025]:
            try:
                bs_code = _to_bs_code(code)
                rs = bs.query_profit_data(bs_code, year=yr, quarter=4)
                if rs.error_code == '0':
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        df = pd.DataFrame(rows, columns=rs.fields)
                        for _, row in df.iterrows():
                            all_records.append({
                                '股票代码': code,
                                '日期': f"{row.get('statDate', '')}",
                                'roe': float(row.get('roeAvg', 0) or 0),
                                'roa': float(row.get('roa', 0) or 0),
                                'gross_margin': float(row.get('grossProfitRatio', 0) or 0),
                                'net_margin': float(row.get('netProfitMargin', 0) or 0),
                            })
            except Exception:
                pass

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def fetch_growth_data(stock_codes):
    """获取成长性: 营收增速, 利润增速"""
    import baostock as bs
    all_records = []
    for code in tqdm(stock_codes, desc="Growth"):
        for yr in [2023, 2024, 2025]:
            try:
                bs_code = _to_bs_code(code)
                rs = bs.query_growth_data(bs_code, year=yr, quarter=4)
                if rs.error_code == '0':
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        df = pd.DataFrame(rows, columns=rs.fields)
                        for _, row in df.iterrows():
                            all_records.append({
                                '股票代码': code,
                                '日期': f"{row.get('statDate', '')}",
                                'revenue_yoy': float(row.get('YOYOperateIncome', 0) or 0),
                                'profit_yoy': float(row.get('YOYNetProfit', 0) or 0),
                            })
            except Exception:
                pass
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def main():
    stock_codes = load_stock_list()
    print(f"Found {len(stock_codes)} stocks")

    if not baostock_login():
        print("Baostock login failed!")
        return

    try:
        # 1. 行业分类
        print("\n[1/3] Fetching industry data...")
        ind_df = fetch_industry_data(stock_codes)
        print(f"  Got {len(ind_df)} industry records")

        # 2. 盈利数据
        print("\n[2/3] Fetching profit data...")
        profit_df = fetch_profit_data(stock_codes[:100])  # 先取前100只
        print(f"  Got {len(profit_df)} profit records")

        # 3. 成长数据
        print("\n[3/3] Fetching growth data...")
        growth_df = fetch_growth_data(stock_codes[:100])
        print(f"  Got {len(growth_df)} growth records")

        # 合并所有数据
        print("\nMerging...")
        merged = None

        if len(profit_df) > 0:
            merged = profit_df.copy()
            merged['日期'] = pd.to_datetime(merged['日期'])

        if len(growth_df) > 0 and merged is not None:
            growth_df['日期'] = pd.to_datetime(growth_df['日期'])
            merged = pd.merge(merged, growth_df, on=['股票代码', '日期'], how='outer')

        if merged is None:
            merged = pd.DataFrame(columns=['股票代码', '日期'])

        # 合并行业数据（横截面）
        if len(ind_df) > 0:
            merged = pd.merge(merged, ind_df, on='股票代码', how='left')

        # 清洁
        for col in merged.columns:
            if col not in ['股票代码', '日期', '行业', '行业分类']:
                merged[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(0)

        merged = merged.drop_duplicates(subset=['股票代码', '日期'], keep='last')
        merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)

        # 保存
        merged.to_csv(CACHE_FILE, index=False)
        print(f"\nSaved {len(merged)} records to {CACHE_FILE}")
        print(f"Columns: {list(merged.columns)}")
        print(f"Coverage: {merged['股票代码'].nunique()} stocks")

        # 检查质量
        for col in ['roe', 'roa', 'gross_margin', 'revenue_yoy', 'profit_yoy']:
            if col in merged.columns:
                nonzero = (merged[col] != 0).sum()
                print(f"  {col}: {nonzero}/{len(merged)} non-zero ({100*nonzero/max(1,len(merged)):.0f}%)")

    finally:
        baostock_logout()


if __name__ == '__main__':
    main()
