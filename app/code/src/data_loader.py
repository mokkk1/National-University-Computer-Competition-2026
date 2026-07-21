"""
data_loader.py — 使用 akshare 获取沪深300成分股的基本面、资金流向、行业等数据

数据来源:
  1. 估值指标 (PE/PB/PS/PCF): ak.stock_a_lg_indicator()
  2. 财务摘要 (ROE/ROA/毛利率/净利率): ak.stock_financial_abstract_ths()
  3. 北向资金: ak.stock_hsgt_holding_analyse_em()
  4. 资金流向 (主力净流入): ak.stock_individual_fund_flow()
  5. 行业分类: ak.stock_board_industry_name_em()

用法:
    python data_loader.py                    # 下载全部数据
    python data_loader.py --update           # 增量更新
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'fundamentals.csv')
CACHE_META = os.path.join(DATA_DIR, 'fundamentals_meta.json')


def load_stock_list():
    """加载沪深300成分股列表，返回 (代码列表, 代码→名称映射)"""
    stock_list_path = os.path.join(DATA_DIR, '..', 'hs300_stock_list.csv')
    if not os.path.exists(stock_list_path):
        # 尝试从 train.csv 获取
        train_path = os.path.join(DATA_DIR, 'train.csv')
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            codes = sorted(df['股票代码'].unique())
            return [str(c) for c in codes], {str(c): str(c) for c in codes}
        raise FileNotFoundError(f"找不到股票列表文件: {stock_list_path}")

    df = pd.read_csv(stock_list_path)
    codes = []
    name_map = {}
    for _, row in df.iterrows():
        code_full = row['code']
        code_name = row['code_name']
        # 提取纯数字代码 (去掉 sh./sz. 前缀)
        if 'sh.' in str(code_full):
            code_num = str(code_full).replace('sh.', '')
        elif 'sz.' in str(code_full):
            code_num = str(code_full).replace('sz.', '')
        else:
            code_num = str(code_full)
        codes.append(code_num)
        name_map[code_num] = code_name
    return sorted(codes), name_map


def safe_fetch(func, name, **kwargs):
    """安全调用 akshare 函数，失败时返回 None"""
    try:
        print(f"  正在获取 {name}...")
        result = func(**kwargs)
        if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
            print(f"    {name} 获取成功: {len(result) if isinstance(result, pd.DataFrame) else 'OK'}")
            return result
        else:
            print(f"    {name} 返回空数据")
            return None
    except Exception as e:
        print(f"    {name} 获取失败: {e}")
        return None


def fetch_valuation_data(stock_codes):
    """
    获取个股估值指标: PE(TTM), PB, PS, PCF, 股息率
    使用 ak.stock_a_lg_indicator()
    """
    try:
        import akshare as ak
    except ImportError:
        print("请安装 akshare: pip install akshare")
        return None

    all_data = []
    for i, code in enumerate(stock_codes):
        try:
            # 判断市场
            if code.startswith('6') or code.startswith('5'):
                symbol = f"sh{code}"
            else:
                symbol = f"sz{code}"

            df = ak.stock_a_lg_indicator(symbol=symbol)
            if df is not None and len(df) > 0:
                df['股票代码'] = code
                # 只保留关键列
                keep_cols = ['trade_date', 'pe', 'pb', 'ps', 'pcf', 'dv_ratio', '股票代码']
                available = [c for c in keep_cols if c in df.columns]
                df = df[available].copy()
                df.rename(columns={'trade_date': '日期'}, inplace=True)
                all_data.append(df)
        except Exception as e:
            pass  # 单只股票失败不影响全局

        if (i + 1) % 50 == 0:
            print(f"  估值数据进度: {i+1}/{len(stock_codes)}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['日期'] = pd.to_datetime(result['日期'])
        return result
    return None


def fetch_financial_data(stock_codes):
    """
    获取财务摘要数据: ROE, ROA, 毛利率, 净利率
    使用同花顺接口
    """
    try:
        import akshare as ak
    except ImportError:
        return None

    all_data = []
    for i, code in enumerate(stock_codes):
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            if df is not None and len(df) > 0:
                # 提取关键指标
                key_indicators = {
                    '净资产收益率(ROE)': 'roe',
                    '总资产报酬率(ROA)': 'roa',
                    '毛利率': 'gross_margin',
                    '净利率': 'net_margin',
                    '营业收入同比增长率': 'revenue_yoy',
                    '归母净利润同比增长率': 'profit_yoy',
                }
                # 数据结构: 行=指标, 列=报告期
                records = []
                for _, row in df.iterrows():
                    indicator = row.iloc[0] if len(row) > 0 else ''
                    eng_name = key_indicators.get(str(indicator))
                    if eng_name:
                        for col in df.columns[1:]:  # 跳过第一列(指标名)
                            val = row[col]
                            if pd.notna(val):
                                records.append({
                                    '日期': pd.to_datetime(str(col)),
                                    '股票代码': code,
                                    eng_name: float(val)
                                })
                if records:
                    fin_df = pd.DataFrame(records)
                    fin_df = fin_df.groupby(['日期', '股票代码'], as_index=False).first()
                    all_data.append(fin_df)
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"  财务数据进度: {i+1}/{len(stock_codes)}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        return result
    return None


def fetch_north_flow_data():
    """
    获取北向资金持股数据
    """
    try:
        import akshare as ak
        df = ak.stock_hsgt_holding_analyse_em()
        if df is not None and len(df) > 0:
            # 提取关键信息: 股票代码、持股比例、持股市值
            keep = ['股票代码', '持股比例', '持股市值']
            available = [c for c in keep if c in df.columns]
            df = df[available].copy()
            # 标准化股票代码
            df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
            df['日期'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            return df
    except Exception as e:
        print(f"  北向资金获取失败: {e}")
    return None


def fetch_fund_flow_data(stock_codes, start_date='20240101', end_date='20260315'):
    """
    获取个股资金流向: 主力净流入、超大单净流入等
    """
    try:
        import akshare as ak
    except ImportError:
        return None

    all_data = []
    for i, code in enumerate(stock_codes):
        try:
            df = ak.stock_individual_fund_flow(
                stock=code,
                market="sh" if (code.startswith('6') or code.startswith('5')) else "sz"
            )
            if df is not None and len(df) > 0:
                df['股票代码'] = code
                keep = ['日期', '股票代码', '主力净流入-净额', '超大单净流入-净额',
                        '大单净流入-净额', '中单净流入-净额', '小单净流入-净额']
                available = [c for c in keep if c in df.columns]
                df = df[available].copy()
                all_data.append(df)
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"  资金流向进度: {i+1}/{len(stock_codes)}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['日期'] = pd.to_datetime(result['日期'])
        return result
    return None


def fetch_industry_data():
    """
    获取行业分类数据
    """
    try:
        import akshare as ak
        # 东方财富行业板块成分股
        df = ak.stock_board_industry_name_em()
        if df is not None and len(df) > 0:
            industry_map = {}
            for _, row in df.iterrows():
                board_name = row.get('板块名称', '')
                board_code = row.get('板块代码', '')
                # 获取该板块的成分股
                try:
                    members = ak.stock_board_industry_cons_em(symbol=board_name)
                    if members is not None and len(members) > 0:
                        for _, m in members.iterrows():
                            code = str(m.get('代码', '')).zfill(6)
                            if code not in industry_map:
                                industry_map[code] = []
                            industry_map[code].append(board_name)
                except Exception:
                    pass
            # 转换为 DataFrame
            records = [{'股票代码': k, '行业': ','.join(v)} for k, v in industry_map.items()]
            return pd.DataFrame(records)
    except Exception as e:
        print(f"  行业分类获取失败: {e}")
    return None


def build_fundamental_cache(force=False):
    """
    构建基本面数据缓存。

    将所有数据源合并为统一的 DataFrame，按日期+股票代码索引。
    使用前向填充处理季度数据的不连续性。
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 检查缓存是否新鲜（7天内）
    if not force and os.path.exists(CACHE_FILE) and os.path.exists(CACHE_META):
        with open(CACHE_META, 'r') as f:
            meta = json.load(f)
        cache_date = datetime.fromisoformat(meta.get('date', '2000-01-01'))
        if (datetime.now() - cache_date).days < 7:
            print(f"缓存有效 ({meta['date']})，跳过下载。使用 --force 强制刷新。")
            return pd.read_csv(CACHE_FILE, parse_dates=['日期'])

    print("=" * 60)
    print("开始获取基本面数据...")
    print("=" * 60)

    stock_codes, name_map = load_stock_list()
    print(f"共有 {len(stock_codes)} 只沪深300成分股")

    # 1. 估值数据
    print("\n[1/5] 获取估值指标...")
    val_df = safe_fetch(fetch_valuation_data, "估值指标", stock_codes=stock_codes[:50])  # 先试50只

    # 2. 财务数据
    print("\n[2/5] 获取财务摘要...")
    fin_df = safe_fetch(fetch_financial_data, "财务摘要", stock_codes=stock_codes[:50])

    # 3. 北向资金
    print("\n[3/5] 获取北向资金持股...")
    north_df = safe_fetch(fetch_north_flow_data, "北向资金")

    # 4. 资金流向
    print("\n[4/5] 获取个股资金流向...")
    flow_df = safe_fetch(fetch_fund_flow_data, "资金流向", stock_codes=stock_codes[:20])

    # 5. 行业分类
    print("\n[5/5] 获取行业分类...")
    ind_df = safe_fetch(fetch_industry_data, "行业分类")

    # 合并所有数据
    print("\n合并数据...")
    merged = None

    # 先合并时序数据 (估值 + 财务 + 资金流向)
    time_series_dfs = [d for d in [val_df, fin_df, flow_df] if d is not None]
    if time_series_dfs:
        merged = time_series_dfs[0]
        for df in time_series_dfs[1:]:
            merged = pd.merge(merged, df, on=['日期', '股票代码'], how='outer')
        merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)
        # 前向填充
        for col in merged.columns:
            if col not in ['日期', '股票代码']:
                merged[col] = merged.groupby('股票代码')[col].ffill()
    else:
        # 至少创建空框架
        merged = pd.DataFrame(columns=['日期', '股票代码'])

    # 合并横截面数据 (北向资金)
    if north_df is not None:
        merged = pd.merge(merged, north_df[['股票代码', '持股比例', '持股市值']],
                          on='股票代码', how='left')

    # 合并行业分类
    if ind_df is not None:
        merge_col = '股票代码'
        if merge_col in ind_df.columns:
            merged = pd.merge(merged, ind_df, on='股票代码', how='left')

    # 填充
    for col in merged.columns:
        if col not in ['日期', '股票代码', '行业'] and merged[col].dtype in ['float64', 'float32']:
            merged[col] = merged[col].fillna(0)
    merged['行业'] = merged.get('行业', '').fillna('')

    # 保存
    merged.to_csv(CACHE_FILE, index=False)
    with open(CACHE_META, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'stocks': len(stock_codes)}, f)

    print(f"\n基本面数据已缓存到: {CACHE_FILE}")
    print(f"  行数: {len(merged):,}")
    print(f"  列数: {len(merged.columns)}")
    print(f"  列名: {list(merged.columns)}")

    return merged


def load_fundamentals():
    """加载基本面缓存数据"""
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, parse_dates=['日期'])
    return None


def main():
    parser = argparse.ArgumentParser(description='获取沪深300基本面数据')
    parser.add_argument('--force', action='store_true', help='强制刷新缓存')
    parser.add_argument('--update', action='store_true', help='增量更新')
    args = parser.parse_args()

    build_fundamental_cache(force=args.force)


if __name__ == '__main__':
    main()
