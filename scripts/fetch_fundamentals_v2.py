"""
fetch_fundamentals_v2.py — 使用 akshare 获取更丰富的基本面数据

数据来源:
  1. stock_financial_abstract_ths() → ROE, ROA, 毛利率, 净利率, 营收增速, 利润增速
  2. stock_a_lg_indicator() → PE, PB, PS, PCF 估值指标
  3. stock_individual_fund_flow() → 主力资金流向

输出: data/fundamentals_v2.csv
"""
import os, sys, json, time
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'fundamentals_v2.csv')
os.makedirs(DATA_DIR, exist_ok=True)


def load_stocks():
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train.csv'))
    return sorted(train['股票代码'].unique().astype(str))


def fetch_financial_ths(code):
    """获取同花顺财务摘要数据"""
    import akshare as ak
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator='按报告期')
        if df is None or len(df) == 0:
            return []
        records = []
        for _, row in df.iterrows():
            try:
                date = pd.to_datetime(row['报告期'])
                records.append({
                    '日期': date,
                    '股票代码': code,
                    'roe': float(str(row.get('净资产收益率(ROE)', 0)).replace('%','') or 0),
                    'roa': float(str(row.get('总资产报酬率(ROA)', 0)).replace('%','') or 0),
                    'gross_margin': float(str(row.get('毛利率', 0)).replace('%','') or 0),
                    'net_margin': float(str(row.get('净利率', 0)).replace('%','') or 0),
                    'revenue_yoy': float(str(row.get('营业收入同比增长率', 0)).replace('%','') or 0),
                    'profit_yoy': float(str(row.get('净利润同比增长率', 0)).replace('%','') or 0),
                    'debt_ratio': float(str(row.get('资产负债率', 0)).replace('%','') or 0),
                })
            except Exception:
                continue
        return records
    except Exception as e:
        return []


def fetch_valuation(code):
    """获取估值指标"""
    import akshare as ak
    try:
        market = 'sh' if (code.startswith('6') or code.startswith('5')) else 'sz'
        symbol = f'{market}{code}'
        df = ak.stock_a_lg_indicator(symbol=symbol)
        if df is None or len(df) == 0:
            return []
        records = []
        for _, row in df.iterrows():
            try:
                records.append({
                    '日期': pd.to_datetime(row['trade_date']),
                    '股票代码': code,
                    'pe': float(row.get('pe', 0) or 0),
                    'pb': float(row.get('pb', 0) or 0),
                    'ps': float(row.get('ps', 0) or 0),
                    'pcf': float(row.get('pcf', 0) or 0),
                })
            except Exception:
                continue
        return records
    except Exception:
        return []


def main():
    stocks = load_stocks()
    print(f"Fetching data for {len(stocks)} stocks...")

    all_fin = []
    all_val = []

    for i, code in enumerate(tqdm(stocks, desc="Financial")):
        fin = fetch_financial_ths(code)
        all_fin.extend(fin)
        val = fetch_valuation(code)
        all_val.extend(val)
        if (i+1) % 20 == 0:
            time.sleep(0.5)  # Rate limiting

    df_fin = pd.DataFrame(all_fin) if all_fin else pd.DataFrame()
    df_val = pd.DataFrame(all_val) if all_val else pd.DataFrame()

    print(f"Financial: {len(df_fin)} records for {df_fin['股票代码'].nunique() if len(df_fin)>0 else 0} stocks")
    print(f"Valuation: {len(df_val)} records for {df_val['股票代码'].nunique() if len(df_val)>0 else 0} stocks")

    # Merge
    if len(df_fin) > 0 and len(df_val) > 0:
        merged = pd.merge(df_fin, df_val, on=['日期', '股票代码'], how='outer')
    elif len(df_fin) > 0:
        merged = df_fin
    else:
        merged = df_val

    if len(merged) > 0:
        merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)
        merged.to_csv(CACHE_FILE, index=False)
        print(f"\nSaved {len(merged)} records to {CACHE_FILE}")
        print(f"Columns: {list(merged.columns)}")
        print(f"Coverage: {merged['股票代码'].nunique()} stocks")
        for col in ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']:
            if col in merged.columns:
                nz = (merged[col] != 0).sum()
                print(f"  {col}: {nz}/{len(merged)} non-zero")
    else:
        print("No data fetched!")


if __name__ == '__main__':
    main()
