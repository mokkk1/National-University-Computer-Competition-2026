"""
self_eval.py — Walk-Forward 自评：模拟官方评测流程

原理：
  官方评测 = 在某个时间点用截止日前的数据训练模型，预测未来5日选股，按实际收益率打分
  Walk-Forward 每个窗口天然就是这个流程的一次独立测试！

方法：
  1. 加载所有 Walk-Forward 窗口的模型
  2. 每个窗口在训练截止日预测 Top-5
  3. 用验证期（截止日后2个月）的实际数据计算5日收益率
  4. 输出各窗口收益率 + 统计摘要

优势：
  - 6 个独立窗口覆盖不同市场环境（震荡、趋势、牛熊）
  - 完全样本外（训练集不包含验证期任何信息）
  - 多窗口统计量（均值、标准差、胜率）比单一自评更可靠

用法：
  python self_eval.py                              # 默认 light 配置
  python self_eval.py --config light               # 指定配置
  python self_eval.py --random-windows 20          # 额外随机20个历史窗口
  python self_eval.py --plot                       # 生成可视化图表
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))

from model import StockTransformer
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from walk_forward import load_walk_forward_models

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BASE = os.path.join(PROJECT_DIR, 'model', 'walk_forward')

# ─── 非季末日预测日期 ─────────────────────────────
# 每个 Walk-Forward 窗口在验证期内取每月中旬的交易日
# 目的：避免季末机构调仓/窗口效应，获取更全面的性能估计
NON_QUARTER_DATES = {
    'window_01': [  # 模型训练截止 2024-09-30, 验证期 10~11月
        ('2024-10-14', 'W1-10月中'),
        ('2024-11-12', 'W1-11月中'),
    ],
    'window_02': [  # 截止 2024-12-31, 验证期 1~2月
        ('2025-01-13', 'W2-1月中'),
        ('2025-02-12', 'W2-2月中'),
    ],
    'window_03': [  # 截止 2025-03-31, 验证期 4~5月
        ('2025-04-14', 'W3-4月中'),
        ('2025-05-12', 'W3-5月中'),
    ],
    'window_04': [  # 截止 2025-06-30, 验证期 7~8月
        ('2025-07-14', 'W4-7月中'),
        ('2025-08-12', 'W4-8月中'),
    ],
    'window_05': [  # 截止 2025-09-30, 验证期 10~11月
        ('2025-10-13', 'W5-10月中'),
        ('2025-11-12', 'W5-11月中'),
    ],
    'window_06': [  # 截止 2025-12-31, 验证期 1~3月
        ('2026-01-12', 'W6-1月中'),
        ('2026-02-12', 'W6-2月中'),
    ],
}


def load_data():
    """加载完整数据集"""
    train = pd.read_csv(os.path.join(PROJECT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(PROJECT_DIR, 'test.csv'))
    full = pd.concat([train, test], ignore_index=True)
    full['日期'] = pd.to_datetime(full['日期'])
    full = full.sort_values(['股票代码', '日期']).reset_index(drop=True)
    return full


def build_features(full_df, feature_num='158+39'):
    """特征工程"""
    feature_engineer = feature_engineer_func_map.get(feature_num, engineer_features_158plus39)
    print(f"  特征工程 ({feature_num})...")
    groups = [g for _, g in full_df.groupby('股票代码', sort=False)]
    processed_list = [feature_engineer(g) for g in tqdm(groups, desc="  特征工程")]
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # ─── 宏观特征拼接 ────────────────────────────
    try:
        from macro_industry import load_macro_features, merge_macro_to_stock
        macro_df = load_macro_features()
        if macro_df is not None:
            processed = merge_macro_to_stock(processed, macro_df)
    except Exception:
        pass

    # ─── 行业分类 ────────────────────────────────
    try:
        from macro_industry import add_industry_features
        processed, _ = add_industry_features(processed)
    except Exception:
        pass
    if 'industry' not in processed.columns:
        processed['industry'] = 0

    # instrument 映射
    all_stocks = sorted(processed['股票代码'].unique())
    stock2idx = {s: i for i, s in enumerate(all_stocks)}
    processed['instrument'] = processed['股票代码'].map(stock2idx)

    feature_cols_template = feature_cloums_map.get(feature_num, feature_cloums_map['158+39'])
    feature_cols = [c for c in feature_cols_template if c in processed.columns]
    # 添加宏观列
    for mc in processed.columns:
        if mc.startswith(('bond', 'north', 'margin', 'usdcny', 'lpr', 'shibor', 'cpi', 'pmi', 'm1', 'm2', 'social')) and mc not in feature_cols:
            feature_cols.append(mc)
    # 添加 industry 列
    if 'industry' in processed.columns and 'industry' not in feature_cols:
        feature_cols.append('industry')

    # ★ 日历特征 — 仅当 scaler 包含时才添加（训练时 use_calendar_features 控制）
    try:
        from walk_forward import add_calendar_features
        processed = add_calendar_features(processed)
        for cc in ['days_to_qe', 'is_qe_month']:
            if cc in processed.columns and cc not in feature_cols:
                feature_cols.append(cc)
    except Exception:
        pass

    return processed, feature_cols, stock2idx


def compute_future_returns(full_df, pred_date, horizon=5):
    """
    计算从 pred_date 后第1天开盘到第5天开盘的真实收益率。

    官方公式: return = (open_day5 - open_day1) / open_day1
    """
    future = full_df[full_df['日期'] > pred_date].copy()
    if future.empty:
        return {}

    returns = {}
    for code, group in future.groupby('股票代码'):
        group = group.sort_values('日期')
        if len(group) >= horizon:
            open_t1 = float(group.iloc[0]['开盘'])
            open_th = float(group.iloc[min(horizon - 1, len(group) - 1)]['开盘'])
            if open_t1 > 1e-4:
                returns[code] = (open_th - open_t1) / open_t1
    return returns


def predict_at_date(model_info, processed_df, feature_cols, stock_codes, pred_date, device, return_aux=False):
    """在指定日期用指定模型预测"""
    cfg = model_info['config']
    seq_len = cfg.get('sequence_length', 60)

    sequences = []
    valid_stocks = []
    for code in stock_codes:
        hist = processed_df[
            (processed_df['股票代码'] == code) &
            (processed_df['日期'] <= pred_date)
        ].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            sequences.append(hist[feature_cols].values.astype(np.float32))
            valid_stocks.append(code)

    if len(sequences) == 0:
        return None

    seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)
    with torch.no_grad():
        if return_aux:
            scores, aux = model_info['model'](seq_tensor, return_aux=True)
            scores_np = scores.squeeze(0).cpu().numpy()
            result = {'scores': scores_np, 'stocks': valid_stocks}
            if 'return_abs' in aux:
                result['return_abs'] = aux['return_abs'].squeeze(0).cpu().numpy()
            return result
        else:
            scores = model_info['model'](seq_tensor)
            scores_np = scores.squeeze(0).cpu().numpy()

    return {'scores': scores_np, 'stocks': valid_stocks}


def ensemble_predict(models, processed_df, feature_cols, stock_codes, pred_date, device):
    """多模型集成预测（median聚合）"""
    all_scores = []
    all_valid_stocks = None

    for m in models:
        result = predict_at_date(m, processed_df, feature_cols, stock_codes, pred_date, device)
        if result is None:
            continue
        if all_valid_stocks is None:
            all_valid_stocks = result['stocks']
        # 确保股票顺序一致
        stock_to_score = dict(zip(result['stocks'], result['scores']))
        scores_aligned = np.array([stock_to_score.get(s, -1e9) for s in all_valid_stocks])
        all_scores.append(scores_aligned)

    if not all_scores:
        return None

    median_scores = np.median(all_scores, axis=0)
    ranked = np.argsort(median_scores)[::-1]

    return {'stocks': all_valid_stocks, 'scores': median_scores, 'ranked': ranked}


def compute_portfolio_return(top_stocks, weights, actual_returns):
    """计算组合加权收益率"""
    total = 0.0
    for code, w in zip(top_stocks, weights):
        ret = actual_returns.get(code, 0.0)
        total += w * ret
    return total


def optimize_weights_simple(scores, top_k=5):
    """简单的 softmax 权重分配"""
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_scores = scores[top_indices]
    # softmax
    exp_scores = np.exp(top_scores - np.max(top_scores))
    weights = exp_scores / exp_scores.sum()
    return top_indices, weights


def evaluate_walk_forward_windows(models, processed_df, feature_cols, full_df, device):
    """
    核心评估：用每个 Walk-Forward 窗口的模型在训练截止日预测，
    用后续真实数据计算收益率。

    V8 改进：集成后处理市场门控 — 根据市场方向动态切换选股策略。
    """
    from market_gate import MarketGate, compute_market_signal

    results = []
    stock_codes = sorted(processed_df['股票代码'].unique())

    # ─── 准备市场门控所需原始数据 ────────────────────────
    # 用原始 full_df 构建涨跌幅、成交量、成交额
    raw_gate = full_df.copy()
    raw_gate = raw_gate.sort_values(['股票代码', '日期'])
    raw_gate['涨跌幅'] = raw_gate.groupby('股票代码')['收盘'].pct_change() * 100
    raw_gate['涨跌幅'] = raw_gate['涨跌幅'].fillna(0)
    # 确保成交量和成交额存在
    for col in ['成交量', '成交额']:
        if col not in raw_gate.columns:
            raw_gate[col] = 0
        else:
            raw_gate[col] = raw_gate[col].fillna(0)
    # 构建每日市场收益率序列
    daily_market = raw_gate.groupby('日期')['涨跌幅'].mean().reset_index()
    daily_market.columns = ['日期', 'market_return']

    for i, m in enumerate(models):
        cfg = m['config']
        window_dir = os.path.basename(m['window_dir'])

        # 从窗口目录名推断训练截止日期
        # ★ 方向二：12 窗口 — 6 季末 + 6 中旬
        window_dates = {
            'window_01': '2024-09-30', 'window_02': '2024-12-31',
            'window_03': '2025-03-31', 'window_04': '2025-06-30',
            'window_05': '2025-09-30', 'window_06': '2025-12-31',
            'window_07': '2024-10-15', 'window_08': '2025-01-15',
            'window_09': '2025-04-15', 'window_10': '2025-07-15',
            'window_11': '2025-10-15', 'window_12': '2026-01-15',
        }
        pred_date_str = window_dates.get(window_dir, None)
        if pred_date_str is None:
            continue

        pred_date = pd.to_datetime(pred_date_str)
        print(f"\n{'='*60}")
        print(f"Window {i+1} ({window_dir}): 预测日期 = {pred_date_str}")
        print(f"{'='*60}")

        # 标准化（使用该窗口的 scaler）
        scaler = m['scaler']

        # 确定 scaler 期望的特征列
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = list(scaler.feature_names_in_)
        else:
            scaler_features = [c for c in feature_cols if c != 'industry']

        eval_df = processed_df.copy()
        # 填充缺失列
        missing = [c for c in scaler_features if c not in eval_df.columns]
        for c in missing:
            eval_df[c] = 0.0
        # 确保 industry 列存在
        if cfg.get('use_industry_embedding') and 'industry' not in eval_df.columns:
            eval_df['industry'] = 0

        # 标准化（排除 industry）
        scaler_cols = [c for c in scaler_features if c in eval_df.columns]
        eval_df[scaler_cols] = eval_df[scaler_cols].replace([np.inf, -np.inf], np.nan)
        eval_df[scaler_cols] = eval_df[scaler_cols].fillna(0)
        eval_df[scaler_cols] = scaler.transform(eval_df[scaler_cols])

        # 模型输入 = scaler_cols + industry
        model_cols = scaler_cols + (['industry'] if cfg.get('use_industry_embedding') else [])

        # 预测（含 aux 输出以获取预测收益率）
        result = predict_at_date(m, eval_df, model_cols, stock_codes, pred_date, device, return_aux=True)
        if result is None:
            print(f"  ❌ 预测失败（无足够数据）")
            continue

        scores = result['scores']
        valid_stocks = result['stocks']
        predicted_returns = result.get('return_abs', None)

        # ─── 计算波动率（从原始收盘价）─────────────────
        volatilities = np.zeros(len(valid_stocks))
        for j, code in enumerate(valid_stocks):
            stock_raw = raw_gate[
                (raw_gate['股票代码'] == code) &
                (raw_gate['日期'] <= pred_date)
            ].sort_values('日期').tail(60)
            if len(stock_raw) >= 20:
                rets = stock_raw['涨跌幅'].values[-20:] / 100
                volatilities[j] = np.std(rets)

        # ─── 市场信号计算 ──────────────────────────────
        recent_mkt = daily_market[daily_market['日期'] <= pred_date].tail(10)
        if len(recent_mkt) >= 2:
            cum_return = (1 + recent_mkt['market_return'] / 100).prod() - 1
        else:
            cum_return = 0.0

        market_signal = {
            'signal': float(cum_return),
            'direction': 1 if cum_return > 0.01 else (-1 if cum_return < -0.01 else 0),
            'confidence': min(0.9, 0.5 + abs(cum_return) * 5),
            'method': 'hs300_return'
        }
        dir_emoji = {1: '📈', -1: '📉', 0: '📊'}
        print(f"  🌐 市场信号: {dir_emoji.get(market_signal['direction'], '❓')} "
              f"direction={market_signal['direction']:+d} "
              f"signal={cum_return:+.3f}")

        # ─── 市场门控选股 ──────────────────────────────
        use_market_gate = cfg.get('use_market_gate', True)
        if use_market_gate:
            gate = MarketGate(strategy='quarter_aware', defensive_weight=0.6)
            top_indices, top_weights = gate.select(
                scores,
                stock_codes=valid_stocks,
                market_signal=market_signal,
                predicted_returns=predicted_returns,
                volatilities=volatilities,
                processed_df=raw_gate,
                pred_date=pred_date,
                top_k=5,
                candidate_k=cfg.get('post_top_k', 10),
                temperature=2.0
            )
            if market_signal['direction'] < 0:
                print(f"  🛡️ 启用防御型选股策略 (defensive_weight=0.6)")
            else:
                print(f"  🚀 启用正常收益门控选股策略")
        else:
            top_indices, top_weights = optimize_weights_simple(scores, top_k=5)

        top_stocks = [valid_stocks[i] for i in top_indices]

        print(f"  Top-5 预测:")
        for rank, (code, w) in enumerate(zip(top_stocks, top_weights)):
            ret_str = ""
            if predicted_returns is not None:
                idx = valid_stocks.index(code) if code in valid_stocks else -1
                if idx >= 0 and idx < len(predicted_returns):
                    ret_str = f"  pred_ret={predicted_returns[idx]:+.3f}"
            print(f"    #{rank+1}: {code}  score={scores[valid_stocks.index(code)]:.4f}"
                  f"  weight={w:.4f}{ret_str}")

        # 计算实际收益率
        actual_returns = compute_future_returns(full_df, pred_date, horizon=5)
        portfolio_ret = compute_portfolio_return(top_stocks, top_weights, actual_returns)

        # 计算 Oracle（理论上限）
        all_rets = {k: v for k, v in actual_returns.items() if not np.isnan(v)}
        oracle_stocks = sorted(all_rets.items(), key=lambda x: x[1], reverse=True)[:5]
        oracle_ret = sum(r for _, r in oracle_stocks) / 5

        print(f"  📊 组合收益率: {portfolio_ret*100:+.2f}%")
        print(f"  📊 Oracle上限:  {oracle_ret*100:+.2f}%")
        if oracle_ret > 0:
            print(f"  📊 相对Oracle:  {portfolio_ret/oracle_ret*100:.1f}%")

        results.append({
            'label': pred_date_str,
            'window': window_dir,
            'pred_date': pred_date_str,
            'portfolio_return': portfolio_ret,
            'oracle_return': oracle_ret,
            'relative_pct': portfolio_ret / oracle_ret * 100 if oracle_ret > 0 else 0,
            'top_stocks': top_stocks,
            'top_weights': list(top_weights),
            'actual_returns': {str(code): actual_returns.get(code, 0) for code in top_stocks},
            'market_direction': market_signal['direction'],
            'market_signal': market_signal['signal'],
            'is_quarter_end': True,
        })

    return results


def evaluate_non_quarter_dates(models, processed_df, feature_cols, full_df, device):
    """
    在每个 Walk-Forward 窗口的验证期内，取非季末日（每月中旬）做预测。
    补充季末自评，得到更全面的性能估计。

    每个窗口的模型在其验证期内预测 2 个非季末日。
    """
    from market_gate import MarketGate

    results = []
    stock_codes = sorted(processed_df['股票代码'].unique())

    # ─── 准备市场门控所需原始数据 ────────────────────────
    raw_gate = full_df.copy()
    raw_gate = raw_gate.sort_values(['股票代码', '日期'])
    raw_gate['涨跌幅'] = raw_gate.groupby('股票代码')['收盘'].pct_change() * 100
    raw_gate['涨跌幅'] = raw_gate['涨跌幅'].fillna(0)
    for col in ['成交量', '成交额']:
        if col not in raw_gate.columns:
            raw_gate[col] = 0
        else:
            raw_gate[col] = raw_gate[col].fillna(0)
    daily_market = raw_gate.groupby('日期')['涨跌幅'].mean().reset_index()
    daily_market.columns = ['日期', 'market_return']

    for m in models:
        cfg = m['config']
        window_dir = os.path.basename(m['window_dir'])

        extra_dates = NON_QUARTER_DATES.get(window_dir, [])
        if not extra_dates:
            continue

        for pred_date_str, label in extra_dates:
            pred_date = pd.to_datetime(pred_date_str)
            # 确保预测日在数据范围内且有后续5个交易日
            available_dates = sorted(full_df['日期'].unique())
            future_dates = [d for d in available_dates if d > pred_date]
            if len(future_dates) < 5:
                print(f"\n  ⚠️ {label} ({pred_date_str}): 后续交易日不足，跳过")
                continue

            print(f"\n{'─'*50}")
            print(f"  {label}: 预测日期 = {pred_date_str}")
            print(f"{'─'*50}")

            # 标准化
            scaler = m['scaler']
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
            else:
                scaler_features = [c for c in feature_cols if c != 'industry']

            eval_df = processed_df.copy()
            missing = [c for c in scaler_features if c not in eval_df.columns]
            for c in missing:
                eval_df[c] = 0.0
            if cfg.get('use_industry_embedding') and 'industry' not in eval_df.columns:
                eval_df['industry'] = 0

            scaler_cols = [c for c in scaler_features if c in eval_df.columns]
            eval_df[scaler_cols] = eval_df[scaler_cols].replace([np.inf, -np.inf], np.nan)
            eval_df[scaler_cols] = eval_df[scaler_cols].fillna(0)
            eval_df[scaler_cols] = scaler.transform(eval_df[scaler_cols])

            model_cols = scaler_cols + (['industry'] if cfg.get('use_industry_embedding') else [])

            # 预测
            pred_result = predict_at_date(m, eval_df, model_cols, stock_codes, pred_date, device, return_aux=True)
            if pred_result is None:
                print(f"    ❌ 预测失败")
                continue

            scores = pred_result['scores']
            valid_stocks = pred_result['stocks']
            predicted_returns = pred_result.get('return_abs', None)

            # 波动率
            volatilities = np.zeros(len(valid_stocks))
            for j, code in enumerate(valid_stocks):
                stock_raw = raw_gate[
                    (raw_gate['股票代码'] == code) &
                    (raw_gate['日期'] <= pred_date)
                ].sort_values('日期').tail(60)
                if len(stock_raw) >= 20:
                    rets = stock_raw['涨跌幅'].values[-20:] / 100
                    volatilities[j] = np.std(rets)

            # 市场信号
            recent_mkt = daily_market[daily_market['日期'] <= pred_date].tail(10)
            if len(recent_mkt) >= 2:
                cum_return = (1 + recent_mkt['market_return'] / 100).prod() - 1
            else:
                cum_return = 0.0

            market_signal = {
                'signal': float(cum_return),
                'direction': 1 if cum_return > 0.01 else (-1 if cum_return < -0.01 else 0),
                'confidence': min(0.9, 0.5 + abs(cum_return) * 5),
                'method': 'hs300_return'
            }
            dir_emoji = {1: '📈', -1: '📉', 0: '📊'}

            # 市场门控选股
            use_market_gate = cfg.get('use_market_gate', True)
            if use_market_gate:
                gate = MarketGate(strategy='quarter_aware', defensive_weight=0.6)
                top_indices, top_weights = gate.select(
                    scores,
                    stock_codes=valid_stocks,
                    market_signal=market_signal,
                    predicted_returns=predicted_returns,
                    volatilities=volatilities,
                    processed_df=raw_gate,
                    pred_date=pred_date,
                    top_k=5,
                    candidate_k=cfg.get('post_top_k', 10),
                    temperature=2.0
                )
                strategy_str = '🛡️防御' if market_signal['direction'] < 0 else '🚀正常'
            else:
                top_indices, top_weights = optimize_weights_simple(scores, top_k=5)
                strategy_str = '📊等权'

            top_stocks = [valid_stocks[i] for i in top_indices]

            # 实际收益率
            actual_returns = compute_future_returns(full_df, pred_date, horizon=5)
            portfolio_ret = compute_portfolio_return(top_stocks, top_weights, actual_returns)

            all_rets = {k: v for k, v in actual_returns.items() if not np.isnan(v)}
            oracle_stocks = sorted(all_rets.items(), key=lambda x: x[1], reverse=True)[:5]
            oracle_ret = sum(r for _, r in oracle_stocks) / 5 if len(oracle_stocks) >= 5 else 0

            marker = '✅' if portfolio_ret > 0 else '❌'
            print(f"    {marker} {strategy_str} | 信号: {dir_emoji.get(market_signal['direction'], '?')} "
                  f"sig={cum_return:+.3f} | 收益: {portfolio_ret*100:+.2f}% "
                  f"(Oracle: {oracle_ret*100:+.2f}%)")

            results.append({
                'label': label,
                'window': window_dir,
                'pred_date': pred_date_str,
                'portfolio_return': portfolio_ret,
                'oracle_return': oracle_ret,
                'relative_pct': portfolio_ret / oracle_ret * 100 if oracle_ret > 0 else 0,
                'top_stocks': top_stocks,
                'top_weights': list(top_weights),
                'actual_returns': {str(code): actual_returns.get(code, 0) for code in top_stocks},
                'market_direction': market_signal['direction'],
                'market_signal': market_signal['signal'],
                'is_quarter_end': False,
            })

    return results
    """
    随机抽取历史日期做测试，评估模型在不同随机时间点的表现。
    使用所有模型的 ensemble 做预测。
    """
    print(f"\n{'='*60}")
    print(f"随机窗口回测 ({n_windows} 个随机日期)")
    print(f"{'='*60}")

    # 获取所有可用的预测日期（需要后续有至少5个交易日）
    all_dates = sorted(full_df['日期'].unique())
    valid_dates = []
    for i, d in enumerate(all_dates):
        if i + 5 < len(all_dates):
            # 确保该日期在训练数据覆盖范围内
            if d >= pd.to_datetime('2024-09-01') and d <= pd.to_datetime('2026-03-06'):
                valid_dates.append(d)

    # 随机采样
    np.random.seed(42)
    sampled = np.random.choice(valid_dates, size=min(n_windows, len(valid_dates)), replace=False)
    sampled = sorted(sampled)

    stock_codes = sorted(processed_df['股票代码'].unique())
    random_results = []

    for pred_date in tqdm(sampled, desc="随机窗口评估"):
        pred_date_str = pred_date.strftime('%Y-%m-%d')

        # 使用第一个模型的 scaler 标准化
        scaler = models[0]['scaler']
        scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else feature_cols
        scaler_cols = [c for c in scaler_features if c in processed_df.columns]

        eval_df = processed_df.copy()
        eval_df[scaler_cols] = eval_df[scaler_cols].replace([np.inf, -np.inf], np.nan)
        eval_df[scaler_cols] = eval_df[scaler_cols].fillna(0)
        eval_df[scaler_cols] = scaler.transform(eval_df[scaler_cols])

        # Ensemble 预测
        result = ensemble_predict(models, eval_df, scaler_cols, stock_codes, pred_date, device)
        if result is None:
            continue

        top_indices, top_weights = optimize_weights_simple(result['scores'], top_k=5)
        top_stocks = [result['stocks'][i] for i in top_indices]

        actual_returns = compute_future_returns(full_df, pred_date, horizon=5)
        portfolio_ret = compute_portfolio_return(top_stocks, top_weights, actual_returns)

        all_rets = {k: v for k, v in actual_returns.items() if not np.isnan(v)}
        oracle_stocks = sorted(all_rets.items(), key=lambda x: x[1], reverse=True)[:5]
        oracle_ret = sum(r for _, r in oracle_stocks) / 5 if len(oracle_stocks) >= 5 else 0

        random_results.append({
            'pred_date': pred_date_str,
            'portfolio_return': portfolio_ret,
            'oracle_return': oracle_ret,
            'relative_pct': portfolio_ret / oracle_ret * 100 if oracle_ret > 0 else 0,
        })

    return random_results


def print_summary(wf_results, non_quarter_results=None, random_results=None):
    """打印评估摘要"""
    print(f"\n{'='*70}")
    print(f"📊 自评摘要")
    print(f"{'='*70}")

    # Walk-Forward 季末汇总
    if wf_results:
        rets = [r['portfolio_return'] for r in wf_results]
        oracle_rets = [r['oracle_return'] for r in wf_results]
        relative = [r['relative_pct'] for r in wf_results if r['relative_pct'] != 0]

        print(f"\n── Walk-Forward 季末窗口回测 ({len(wf_results)} 个窗口) ──")
        print(f"  各窗口收益率:")
        for r in wf_results:
            marker = "✅" if r['portfolio_return'] > 0 else "❌"
            label = r.get('label', r['pred_date'])
            print(f"    {marker} {label}: {r['portfolio_return']*100:+.2f}% "
                  f"(Oracle: {r['oracle_return']*100:+.2f}%, "
                  f"相对: {r['relative_pct']:.0f}%)")

        print(f"\n  季末统计:")
        print(f"    平均收益率:     {np.mean(rets)*100:+.2f}%")
        print(f"    中位数收益率:   {np.median(rets)*100:+.2f}%")
        print(f"    标准差:         {np.std(rets)*100:.2f}%")
        print(f"    最大值:         {np.max(rets)*100:+.2f}%")
        print(f"    最小值:         {np.min(rets)*100:+.2f}%")
        print(f"    正收益窗口:     {sum(1 for r in rets if r > 0)}/{len(rets)} "
              f"({sum(1 for r in rets if r > 0)/len(rets)*100:.0f}%)")

        if len(rets) >= 3:
            sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 / 5)
            print(f"    年化夏普(approx): {sharpe:.2f}")

    # 非季末日汇总
    if non_quarter_results:
        nq_rets = [r['portfolio_return'] for r in non_quarter_results]

        print(f"\n── 非季末日回测 ({len(non_quarter_results)} 个日期) ──")
        print(f"  各日期收益率:")
        for r in non_quarter_results:
            marker = "✅" if r['portfolio_return'] > 0 else "❌"
            label = r.get('label', r['pred_date'])
            print(f"    {marker} {label}: {r['portfolio_return']*100:+.2f}% "
                  f"(Oracle: {r['oracle_return']*100:+.2f}%)")

        print(f"\n  非季末统计:")
        print(f"    平均收益率:     {np.mean(nq_rets)*100:+.2f}%")
        print(f"    中位数收益率:   {np.median(nq_rets)*100:+.2f}%")
        print(f"    标准差:         {np.std(nq_rets)*100:.2f}%")
        print(f"    最大值:         {np.max(nq_rets)*100:+.2f}%")
        print(f"    最小值:         {np.min(nq_rets)*100:+.2f}%")
        print(f"    正收益比例:     {sum(1 for r in nq_rets if r > 0)}/{len(nq_rets)} "
              f"({sum(1 for r in nq_rets if r > 0)/len(nq_rets)*100:.0f}%)")

        if len(nq_rets) >= 3:
            sharpe = np.mean(nq_rets) / (np.std(nq_rets) + 1e-8) * np.sqrt(252 / 5)
            print(f"    年化夏普(approx): {sharpe:.2f}")
    else:
        nq_rets = []

    # 随机窗口汇总
    if random_results:
        r_rets = [r['portfolio_return'] for r in random_results]
        r_relative = [r['relative_pct'] for r in random_results if r['relative_pct'] != 0]

        print(f"\n── 随机窗口回测 ({len(random_results)} 个日期) ──")
        print(f"    平均收益率:     {np.mean(r_rets)*100:+.2f}%")
        print(f"    中位数收益率:   {np.median(r_rets)*100:+.2f}%")
        print(f"    标准差:         {np.std(r_rets)*100:.2f}%")
        print(f"    正收益概率:     {sum(1 for r in r_rets if r > 0)/len(r_rets)*100:.0f}%")
        print(f"    相对Oracle均值: {np.mean(r_relative):.1f}%")

    # 综合结论
    all_rets = []
    if wf_results:
        all_rets.extend([r['portfolio_return'] for r in wf_results])
    if non_quarter_results:
        all_rets.extend([r['portfolio_return'] for r in non_quarter_results])
    if random_results:
        all_rets.extend([r['portfolio_return'] for r in random_results])

    if all_rets:
        total_tests = (len(wf_results) if wf_results else 0) + \
                      (len(non_quarter_results) if non_quarter_results else 0) + \
                      (len(random_results) if random_results else 0)
        print(f"\n── 综合评估 ({total_tests} 次独立测试) ──")
        print(f"    季末日均值:    {np.mean([r['portfolio_return'] for r in wf_results])*100:+.2f}%"
              if wf_results else "")
        if non_quarter_results:
            print(f"    非季末日均值:  {np.mean([r['portfolio_return'] for r in non_quarter_results])*100:+.2f}%")
        print(f"    全部均值:      {np.mean(all_rets)*100:+.2f}%")
        print(f"    全部正收益:    {sum(1 for r in all_rets if r > 0)}/{len(all_rets)} "
              f"({sum(1 for r in all_rets if r > 0)/len(all_rets)*100:.0f}%)")
        print(f"    期望年化:      {np.mean(all_rets) * 252 / 5 * 100:.1f}%")

        # 季末 vs 非季末对比
        if wf_results and non_quarter_results:
            qe_mean = np.mean([r['portfolio_return'] for r in wf_results])
            nq_mean = np.mean([r['portfolio_return'] for r in non_quarter_results])
            diff = nq_mean - qe_mean
            print(f"\n  🔍 季末效应分析:")
            print(f"     季末日均值:  {qe_mean*100:+.2f}%")
            print(f"     非季末日均值: {nq_mean*100:+.2f}%")
            print(f"     差异:         {diff*100:+.2f}pp {'← 季末拖累' if diff > 0 else '← 季末反更好'}")

        # 与之前 V7 对比
        print(f"\n  💡 对比参考:")
        print(f"     V7 自评 (固定窗口): 6.15%")
        print(f"     V7 官方得分:        -1.29%")
        print(f"     V8 Enhanced 均值:   -1.37%")
        print(f"     本次综合评估:       {np.mean(all_rets)*100:+.2f}%")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward 自评')
    parser.add_argument('--config', type=str, default='light', help='模型配置')
    parser.add_argument('--seeds', type=str, default='42', help='随机种子')
    parser.add_argument('--wf-dir', type=str, default=MODEL_BASE, help='模型目录')
    parser.add_argument('--random-windows', type=int, default=0,
                        help='额外随机窗口数量 (0=不启用)')
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 路径')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    print("\n加载 Walk-Forward 模型...")
    models = load_walk_forward_models(args.wf_dir, args.config, seeds, device)
    print(f"加载了 {len(models)} 个模型")

    if len(models) == 0:
        print("❌ 没有模型！请先运行 run_optimized.py --train-only")
        sys.exit(1)

    # 加载数据 + 特征工程
    print("\n加载数据...")
    full_df = load_data()
    first_cfg = models[0]['config']
    feature_num = first_cfg.get('feature_num', '158+39')
    processed_df, feature_cols, stock2idx = build_features(full_df, feature_num)

    # Walk-Forward 窗口评估（季末日）
    print(f"\n{'='*70}")
    print(f"Walk-Forward 季末窗口回测")
    print(f"{'='*70}")
    wf_results = evaluate_walk_forward_windows(
        models, processed_df, feature_cols, full_df, device
    )

    # 非季末日评估
    print(f"\n\n{'='*70}")
    print(f"非季末日回测 (每月中旬)")
    print(f"{'='*70}")
    non_quarter_results = evaluate_non_quarter_dates(
        models, processed_df, feature_cols, full_df, device
    )

    # 随机窗口评估
    random_results = None
    if args.random_windows > 0:
        random_results = evaluate_random_windows(
            processed_df, feature_cols, full_df, models, device,
            n_windows=args.random_windows
        )

    # 打印摘要
    print_summary(wf_results, non_quarter_results, random_results)

    # 保存
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(args.wf_dir, f'self_eval_{ts}.json')

    output_data = {
        'config': args.config,
        'seeds': seeds,
        'walk_forward_results': wf_results,
        'non_quarter_results': non_quarter_results,
        'random_results': random_results,
        'timestamp': datetime.now().isoformat(),
    }
    with open(args.output, 'w') as f:
        # 处理 numpy 类型
        def convert(obj):
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(convert(output_data), f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果保存到: {args.output}")


if __name__ == '__main__':
    main()
