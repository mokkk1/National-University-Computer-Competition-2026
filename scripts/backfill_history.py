#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补拉 2010-2023 历史行情数据（baostock 后复权），与现有 stock_data.csv (2024+) 合并。

后复权价格锚定上市日，补拉的早期数据与现有 2024+ 数据在边界处天然连续。
注意：使用当前成分股列表拉取全历史，存在幸存者偏差（研究阶段可接受）。

用法:
  python backfill_history.py                # 拉取（断点续传）+ 合并 + 重新划分 train/test
  python backfill_history.py --merge-only   # 只做合并与划分（拉取已完成时）
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'code'))
from get_stock_data import login, logout, get_stock_history  # noqa: E402

STOCK_LIST = os.path.join(BASE, 'hs300_stock_list.csv')
STOCK_DATA = os.path.join(BASE, 'stock_data.csv')
BACKFILL = os.path.join(BASE, 'data_backfill_2010_2023.csv')
EMPTY_LIST = os.path.join(BASE, 'data_backfill_empty.txt')
BACKUP_DIR = os.path.join(BASE, 'backup_before_2010_extension')

START_DATE = '2010-01-01'
END_DATE = '2023-12-31'


def load_done_set():
    """已完成的股票（有数据的 + 确认无早期数据的）"""
    done = set()
    if os.path.exists(BACKFILL):
        done |= set(pd.read_csv(BACKFILL, dtype={'股票代码': str},
                                usecols=['股票代码'])['股票代码'].unique())
    if os.path.exists(EMPTY_LIST):
        with open(EMPTY_LIST) as f:
            done |= {line.strip() for line in f if line.strip()}
    return done


def fetch():
    stocks = pd.read_csv(STOCK_LIST)
    done = load_done_set()
    if done:
        print(f"断点续传: 已完成 {len(done)}/{len(stocks)} 只", flush=True)

    login()
    try:
        ok, fail = 0, 0
        for i, row in stocks.iterrows():
            bs_code = row['code']
            pure = bs_code.split('.')[-1].zfill(6)
            if pure in done:
                continue
            try:
                df = get_stock_history(bs_code, START_DATE, END_DATE)
                if df is not None and not df.empty:
                    df.to_csv(BACKFILL, mode='a',
                              header=not os.path.exists(BACKFILL),
                              index=False, encoding='utf-8-sig')
                    ok += 1
                    print(f"[{i+1}/{len(stocks)}] {bs_code} {row.get('code_name', '')} "
                          f"+{len(df)} 行 ({df['日期'].min()} ~ {df['日期'].max()})", flush=True)
                else:
                    with open(EMPTY_LIST, 'a') as f:
                        f.write(pure + '\n')
                    print(f"[{i+1}/{len(stocks)}] {bs_code} 无 {END_DATE} 前数据（晚于该区间上市）", flush=True)
            except Exception as e:
                fail += 1
                print(f"[{i+1}/{len(stocks)}] {bs_code} 失败: {e}", flush=True)
            if ok > 0 and ok % 20 == 0:
                time.sleep(1)
        print(f"拉取完成: 成功 {ok}, 失败 {fail}", flush=True)
        return fail == 0
    finally:
        logout()


def merge_and_split():
    # 备份原数据（仅首次，不覆盖已有备份）
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in ['stock_data.csv', 'train.csv', 'test.csv']:
        src, dst = os.path.join(BASE, f), os.path.join(BACKUP_DIR, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"备份: {f} -> {BACKUP_DIR}")

    old = pd.read_csv(STOCK_DATA, dtype={'股票代码': str})
    new = pd.read_csv(BACKFILL, dtype={'股票代码': str})
    old['日期'] = pd.to_datetime(old['日期']).dt.strftime('%Y-%m-%d')
    new['日期'] = pd.to_datetime(new['日期']).dt.strftime('%Y-%m-%d')

    # 安全网：backfill 只保留早于现有数据起点的部分，避免边界重叠
    old_min = old['日期'].min()
    new = new[new['日期'] < old_min]

    merged = pd.concat([new, old], ignore_index=True)
    merged = merged.sort_values(['股票代码', '日期']).reset_index(drop=True)
    merged.to_csv(STOCK_DATA, index=False, encoding='utf-8-sig')
    print(f"合并完成: {len(merged):,} 行, {merged['股票代码'].nunique()} 只股票, "
          f"{merged['日期'].min()} ~ {merged['日期'].max()}", flush=True)

    # 重新划分 train/test（test 窗口保持不变）
    subprocess.run([sys.executable, os.path.join(BASE, 'split_train_test.py'),
                    '--input', STOCK_DATA, '--output-dir', BASE,
                    '--train-start', '2010-01-04', '--train-end', '2026-03-06'],
                   check=True, cwd=BASE)


def main():
    parser = argparse.ArgumentParser(description='补拉 2010-2023 历史数据并合并')
    parser.add_argument('--merge-only', action='store_true', help='跳过拉取，只做合并与划分')
    args = parser.parse_args()

    if not args.merge_only:
        stocks_total = len(pd.read_csv(STOCK_LIST))
        # baostock 长会话可能被服务端断开，自动重新登录续传（最多5轮）
        for attempt in range(1, 6):
            fetch()
            remaining = stocks_total - len(load_done_set())
            if remaining == 0:
                break
            print(f"第 {attempt} 轮后仍有 {remaining} 只未完成，15 秒后重新登录续传...", flush=True)
            time.sleep(15)
        remaining = stocks_total - len(load_done_set())
        if remaining > 0:
            print(f"⚠️ 经多轮重试仍有 {remaining} 只失败，请稍后重跑本脚本；未执行合并", flush=True)
            sys.exit(1)
    merge_and_split()


if __name__ == '__main__':
    main()
