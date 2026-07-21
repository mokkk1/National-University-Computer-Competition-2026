"""
predict.py — 使用训练好的模型生成 Top-5 选股结果

用法：
    python predict.py                          # 预测 test.csv 最新日期的 Top-5
    python predict.py --date 2026-03-13        # 预测指定日期
    python predict.py --top-k 10               # 输出 Top-10
    python predict.py --multi-period 5         # 多周期预测融合(最近5日)
    python predict.py --ensemble               # 使用集成模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import os
import sys
import json
from tqdm import tqdm

# 添加 src 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))

from model import StockTransformer, LightweightStockRanker
from utils import engineer_features_158plus39, optimize_weights, select_top_stocks_with_gate
from train import feature_cloums_map, feature_engineer_func_map
from market_gate import MarketGate, compute_market_signal

# ─── 配置 ───────────────────────────────────────────────
# 路径：优先使用环境变量，回退到基于脚本位置的相对路径
_PROJECT_ROOT = os.environ.get(
    'CSI300_PROJECT_DIR',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODEL_DIR = os.environ.get(
    'CSI300_MODEL_DIR',
    os.path.join(_PROJECT_ROOT, 'model', 'walk_forward_v8_2021', 'W6')
)
DATA_DIR = os.environ.get('CSI300_DATA_DIR', _PROJECT_ROOT)

# 特征列（158+39，不含 instrument）
FEATURE_39_COLS = [
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
    'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
    'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]

BASE_COLS = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']


def load_model_and_scaler(model_dir):
    """加载最佳模型、scaler 和配置"""
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    model_path = os.path.join(model_dir, 'best_model.pth')

    # 需要先确定 input_dim，用临时数据跑一次特征工程拿到特征数
    # 这里直接用 config 推断: 158+39 = 197 维，减去 instrument(1) + base_cols(10) = 197 不对
    # 实际特征列数需要从 scaler 的 n_features_in_ 获取
    input_dim = scaler.n_features_in_

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 根据配置选择模型类
    use_model = config.get('use_model', 'transformer')
    if use_model == 'lightweight':
        model = LightweightStockRanker(input_dim=input_dim, config=config, num_stocks=300)
    else:
        model = StockTransformer(input_dim=input_dim, config=config, num_stocks=300)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    print(f"✅ 模型加载成功: {model_path}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   输入维度: {input_dim}")
    print(f"   设备: {device}")

    return model, scaler, config, device


def load_and_preprocess_data(data_dir, scaler, config):
    """加载训练+测试数据，运行特征工程（使用与训练一致的配置），标准化"""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    print(f"\n📂 加载数据...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"   训练集: {train_df.shape[0]:,} 行, 日期范围 {train_df['日期'].min()} ~ {train_df['日期'].max()}")
    print(f"   测试集: {test_df.shape[0]:,} 行, 日期范围 {test_df['日期'].min()} ~ {test_df['日期'].max()}")

    # 合并
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 特征工程（使用训练配置中的 feature_num）
    feature_num = config.get('feature_num', '158+39')
    feature_engineer = feature_engineer_func_map.get(feature_num, engineer_features_158plus39)
    print(f"\n🔧 运行特征工程 ({feature_num})...")
    groups = [group for _, group in full_df.groupby('股票代码', sort=False)]
    processed_list = []
    for group in tqdm(groups, desc="特征工程"):
        processed = feature_engineer(group)
        processed_list.append(processed)

    processed_df = pd.concat(processed_list).reset_index(drop=True)
    processed_df['日期'] = pd.to_datetime(processed_df['日期'])

    # 构建 stock → idx 映射（与训练时一致：sorted(all_stock_ids)）
    all_stock_ids = processed_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    processed_df['instrument'] = processed_df['股票代码'].map(stockid2idx)

    # 确定特征列（使用训练时的 feature_cloums_map）
    all_feature_cols_template = feature_cloums_map.get(feature_num, feature_cloums_map['158+39'])
    all_feature_cols = [c for c in all_feature_cols_template if c in processed_df.columns]
    missing = [c for c in all_feature_cols_template if c not in processed_df.columns]
    if missing:
        print(f"⚠️  缺失特征列 ({len(missing)}): {missing[:10]}...")

    print(f"   使用 {len(all_feature_cols)} 个特征列 (scaler 期望 {scaler.n_features_in_})")

    # ⚠️ 保存原始收盘价用于波动率计算（必须在标准化之前）
    raw_close = processed_df[['股票代码', '日期', '收盘']].copy()

    # 标准化（用训练时的 scaler）
    print(f"\n📏 标准化...")
    processed_df[all_feature_cols] = processed_df[all_feature_cols].replace([np.inf, -np.inf], np.nan)
    processed_df[all_feature_cols] = processed_df[all_feature_cols].fillna(0)
    processed_df[all_feature_cols] = scaler.transform(processed_df[all_feature_cols])

    return processed_df, all_feature_cols, stockid2idx, raw_close


def predict_for_date(data, features, stock_codes, date, model, scaler, config, device):
    """对指定日期预测所有股票的排序分数 + 绝对收益率"""
    sequence_length = config['sequence_length']

    sequences = []
    valid_stocks = []

    for code in stock_codes:
        stock_hist = data[
            (data['股票代码'] == code) &
            (data['日期'] <= date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_hist) == sequence_length:
            seq = stock_hist[features].values.astype(np.float32)
            sequences.append(seq)
            valid_stocks.append(code)

    if len(sequences) == 0:
        print(f"❌ 日期 {date} 没有足够的数据（需要 {sequence_length} 天历史）")
        return None

    # [1, num_stocks, seq_len, features]
    seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)

    with torch.no_grad():
        scores, aux = model(seq_tensor, return_aux=True)  # 获取排序分数 + 辅助输出
        scores = scores.squeeze(0).cpu().numpy()  # [num_stocks]
        predicted_returns = aux['return_abs'].squeeze(0).cpu().numpy()  # 绝对收益率 [num_stocks]

    results = pd.DataFrame({
        '股票代码': valid_stocks,
        '预测分数': scores,
        '预测收益率': predicted_returns
    })
    results = results.sort_values('预测分数', ascending=False).reset_index(drop=True)
    results['排名'] = range(1, len(results) + 1)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Top-K 选股预测')
    parser.add_argument('--date', type=str, default=None,
                        help='预测日期 (YYYY-MM-DD)，默认使用 test.csv 最新日期')
    parser.add_argument('--top-k', type=int, default=5,
                        help='输出 Top-K 股票 (默认 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 CSV 路径 (默认保存到模型目录)')
    parser.add_argument('--multi-period', type=int, default=0,
                        help='多周期预测融合: 使用最近N个日期预测的平均分数 (阶段5)')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用集成模型预测 (阶段4)')
    args = parser.parse_args()

    # 1. 加载模型
    model, scaler, config, device = load_model_and_scaler(MODEL_DIR)

    # 2. 加载并预处理数据
    processed_df, feature_cols, stockid2idx, raw_close = load_and_preprocess_data(DATA_DIR, scaler, config)

    # 3. 确定预测日期
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    test_dates = sorted(test_df['日期'].unique())

    if args.date:
        pred_date = pd.to_datetime(args.date)
    else:
        pred_date = test_dates[-1]

    print(f"\n🎯 预测日期: {pred_date.strftime('%Y-%m-%d')}")

    # 4. 获取该日期的股票列表
    day_stocks = test_df[test_df['日期'] == pred_date]['股票代码'].unique()
    # 也可以用全部历史中出现的股票
    all_stocks = processed_df['股票代码'].unique()
    print(f"   候选股票数: {len(all_stocks)}（全部历史）/ {len(day_stocks)}（当日有数据的）")

    # 5. 预测
    stock_codes = sorted(all_stocks)  # 用全部股票，保证足够数量

    # ─── 阶段5: 多周期预测融合 ──────────────────────
    if args.multi_period > 1:
        print(f"\nMulti-period prediction: averaging over last {args.multi_period} dates")
        # 获取最近的N个预测日期
        test_dates_for_pred = test_dates[-args.multi_period:] if len(test_dates) >= args.multi_period else test_dates
        print(f"  Prediction dates: {[d.strftime('%Y-%m-%d') for d in test_dates_for_pred]}")

        all_pred_results = []
        for d in test_dates_for_pred:
            day_results = predict_for_date(processed_df, feature_cols, stock_codes,
                                           d, model, scaler, config, device)
            if day_results is not None:
                all_pred_results.append(day_results.set_index('股票代码')['预测分数'])

        if all_pred_results:
            # 平均所有日期的分数
            avg_scores = pd.concat(all_pred_results, axis=1).mean(axis=1)
            results = pd.DataFrame({
                '股票代码': avg_scores.index,
                '预测分数': avg_scores.values
            }).sort_values('预测分数', ascending=False).reset_index(drop=True)
            results['排名'] = range(1, len(results) + 1)
            print(f"  Averaged predictions from {len(all_pred_results)} dates")
        else:
            print("  Multi-period prediction failed, falling back to single-date")
            results = predict_for_date(processed_df, feature_cols, stock_codes,
                                       pred_date, model, scaler, config, device)
    else:
        results = predict_for_date(processed_df, feature_cols, stock_codes,
                                   pred_date, model, scaler, config, device)

    if results is None:
        sys.exit(1)

    # 6. 输出 Top-K（使用市场门控 + 收益门控 + Sharpe-like 评分）
    top_k = args.top_k

    # 计算波动率用于后处理（使用原始收盘价，非标准化值）
    volatilities = None
    if config.get('use_volatility_penalty', True):
        vols = []
        for code in stock_codes:
            stock_data = raw_close[raw_close['股票代码'] == code].sort_values('日期')
            if len(stock_data) >= 20:
                close_prices = stock_data['收盘'].values.astype(float)
                returns = np.diff(close_prices) / (close_prices[:-1] + 1e-12)
                vols.append(np.std(returns[-20:]))
            else:
                vols.append(0.0)
        volatilities = np.array(vols)

    # ─── 市场门控后处理（P0改进）────────────────────
    # 根据当前市场环境动态调整选股策略
    use_market_gate = config.get('use_market_gate', True)
    if use_market_gate:
        # ⚠️ 使用原始收盘价计算市场信号（不能用标准化后的数据）
        # 构建临时的原始数据DataFrame用于市场信号计算
        raw_for_signal = raw_close.copy()
        # 计算每只股票每天的收益率（从原始收盘价）
        raw_for_signal = raw_for_signal.sort_values(['股票代码', '日期'])
        raw_for_signal['return_1d'] = raw_for_signal.groupby('股票代码')['收盘'].pct_change()
        # 聚合成每日市场均值
        daily_market = raw_for_signal.groupby('日期')['return_1d'].mean().reset_index()
        daily_market.columns = ['日期', 'market_return']
        daily_market = daily_market.dropna()

        # 计算近10日累计涨跌
        recent = daily_market[daily_market['日期'] <= pred_date].tail(10)
        if len(recent) >= 2:
            cum_return = (1 + recent['market_return']).prod() - 1
        else:
            cum_return = 0.0

        market_signal = {
            'signal': float(cum_return),
            'direction': 1 if cum_return > 0.01 else (-1 if cum_return < -0.01 else 0),
            'confidence': min(0.9, 0.5 + abs(cum_return) * 5),
            'method': 'hs300_return'
        }
        direction_emoji = {1: '📈', -1: '📉', 0: '📊'}
        print(f"\n🌐 市场信号: {direction_emoji.get(market_signal['direction'], '❓')} "
              f"direction={market_signal['direction']:+d} "
              f"signal={market_signal['signal']:+.3f} "
              f"(近10日HS300累计涨跌)")

        # 准备防御性选股所需数据（使用原始close计算涨跌幅）
        eval_for_gate = raw_close.copy()
        eval_for_gate = eval_for_gate.sort_values(['股票代码', '日期'])
        eval_for_gate['涨跌幅'] = eval_for_gate.groupby('股票代码')['收盘'].pct_change() * 100
        eval_for_gate['涨跌幅'] = eval_for_gate['涨跌幅'].fillna(0)
        # 合并成交量数据（从processed_df取，注意是标准化后的，仅用于防御性评分估算）
        if '成交量' in processed_df.columns:
            vol_df = processed_df[['股票代码', '日期', '成交量']].copy()
            eval_for_gate = eval_for_gate.merge(vol_df, on=['股票代码', '日期'], how='left')
            eval_for_gate['成交量'] = eval_for_gate['成交量'].fillna(0)
        else:
            eval_for_gate['成交量'] = 0
        if '成交额' in processed_df.columns:
            amt_df = processed_df[['股票代码', '日期', '成交额']].copy()
            eval_for_gate = eval_for_gate.merge(amt_df, on=['股票代码', '日期'], how='left')
            eval_for_gate['成交额'] = eval_for_gate['成交额'].fillna(0)
        else:
            eval_for_gate['成交额'] = 0

        gate = MarketGate(strategy='adaptive', defensive_weight=0.6)
        top_indices, top_weights = gate.select(
            results['预测分数'].values,
            stock_codes=stock_codes,
            market_signal=market_signal,
            predicted_returns=results['预测收益率'].values,
            volatilities=volatilities,
            processed_df=eval_for_gate,
            pred_date=pred_date,
            top_k=top_k,
            candidate_k=config.get('post_top_k', 10),
            temperature=2.0
        )

        if market_signal['direction'] < 0:
            print(f"  🛡️ 启用防御型选股策略 (defensive_weight=0.6)")
        else:
            print(f"  🚀 启用正常收益门控选股策略")
    else:
        # 使用原有的收益门控选股（绝对收益过滤 + Sharpe-like评分）
        if config.get('use_return_gate', True):
            top_indices, top_weights = select_top_stocks_with_gate(
                results['预测分数'].values,
                predicted_returns=results['预测收益率'].values,
                volatilities=volatilities,
                top_k=top_k,
                candidate_k=config.get('post_top_k', 10),
                min_return_threshold=config.get('min_return_threshold', 0.0),
                temperature=2.0,
                fallback=config.get('return_gate_fallback', 'equal')
            )
        else:
            top_indices, top_weights = optimize_weights(
                results['预测分数'].values,
                volatilities=volatilities,
                top_k=top_k,
                candidate_k=config.get('post_top_k', 10),
                use_volatility_penalty=config.get('use_volatility_penalty', True),
                temperature=2.0
            )

    top_results = results.iloc[top_indices].copy()
    top_results['权重'] = top_weights
    top_results['排名'] = range(1, len(top_results) + 1)

    # 尝试匹配股票名称
    stock_list_path = os.path.join(DATA_DIR, 'hs300_stock_list.csv')
    if os.path.exists(stock_list_path):
        stock_names = pd.read_csv(stock_list_path)
        stock_names['code_num'] = stock_names['code'].str.extract(r'\.(\d+)').astype(str)
        # test.csv 里的股票代码是纯数字，需要匹配
        top_results['股票名称'] = top_results['股票代码'].astype(str).map(
            stock_names.set_index('code_num')['code_name']
        )

    print(f"\n{'='*60}")
    print(f"  📊 Top-{top_k} 选股结果 — {pred_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    print(top_results.to_string(index=False))
    print(f"{'='*60}")

    # 7. 保存
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(MODEL_DIR, f'top{top_k}_{pred_date.strftime("%Y%m%d")}.csv')

    top_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存到: {output_path}")

    # 同时保存竞赛标准格式
    comp_format = top_results[['股票代码', '权重']].copy()
    comp_format.columns = ['stock_id', 'weight']
    comp_output = os.path.join(MODEL_DIR, f'result_{pred_date.strftime("%Y%m%d")}.csv')
    comp_format.to_csv(comp_output, index=False, encoding='utf-8-sig')
    print(f"💾 竞赛格式已保存到: {comp_output}")

    # 同时保存全量预测结果
    full_output = os.path.join(MODEL_DIR, f'full_predictions_{pred_date.strftime("%Y%m%d")}.csv')
    results.to_csv(full_output, index=False, encoding='utf-8-sig')
    print(f"💾 全量预测已保存到: {full_output}")

    return top_results


if __name__ == '__main__':
    main()
