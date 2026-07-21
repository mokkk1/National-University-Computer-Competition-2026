"""
score_self.py — 自评脚本：模拟竞赛评分流程

竞赛评分公式:
  Score = Σ(weight_i × return_i)
  其中 return_i = (open_day5 - open_day1) / open_day1

流程:
  1. 使用训练数据 (train.csv) 训练模型 / 加载已有模型
  2. 在训练截止日 (2026-03-06) 进行预测
  3. 用测试数据 (2026-03-09 ~ 2026-03-13) 计算实际5日收益率
  4. 加权求和得到自评得分
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))

from model import StockTransformer
from utils import engineer_features_158plus39, optimize_weights, select_top_stocks_with_gate
from train import feature_cloums_map, feature_engineer_func_map

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', '60_158+39+fundamental+momentum_v7')
DATA_DIR = os.path.dirname(__file__)


def load_model():
    """加载已训练的模型"""
    with open(os.path.join(MODEL_DIR, 'config.json')) as f:
        config = json.load(f)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    input_dim = scaler.n_features_in_

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(input_dim=input_dim, config=config, num_stocks=300)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'best_model.pth'),
        map_location=device, weights_only=True
    ), strict=False)
    model.to(device)
    model.eval()
    return model, scaler, config, device


def prepare_features(config):
    """加载数据、特征工程（使用与训练一致的配置）"""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    print(f"Train: {train.shape[0]:,} rows, {train['日期'].min()} ~ {train['日期'].max()}")
    print(f"Test:  {test.shape[0]:,} rows, {test['日期'].min()} ~ {test['日期'].max()}")

    full = pd.concat([train, test], ignore_index=True)
    full['日期'] = pd.to_datetime(full['日期'])
    full = full.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 特征工程（使用训练配置中的 feature_num）
    feature_num = config.get('feature_num', '158+39')
    feature_engineer = feature_engineer_func_map.get(feature_num, engineer_features_158plus39)
    print(f"Running feature engineering ({feature_num})...")
    groups = [g for _, g in full.groupby('股票代码', sort=False)]
    processed_list = []
    for g in groups:
        processed_list.append(feature_engineer(g))
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # 股票索引
    all_stocks = sorted(processed['股票代码'].unique())
    stock2idx = {s: i for i, s in enumerate(all_stocks)}
    processed['instrument'] = processed['股票代码'].map(stock2idx)

    # 获取特征列（与训练时一致）
    all_fcols = feature_cloums_map.get(feature_num, feature_cloums_map['158+39'])
    feature_cols = [c for c in all_fcols if c in processed.columns]
    expected_dim = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl')).n_features_in_
    print(f"Features: {len(feature_cols)} columns (scaler expects {expected_dim})")

    return processed, feature_cols, all_stocks, train, test


def compute_actual_returns(test_df):
    """用测试集计算每只股票的实际5日收益率"""
    test_df = test_df.copy()
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    returns = {}
    for code in test_df['股票代码'].unique():
        sd = test_df[test_df['股票代码'] == code].sort_values('日期')
        if len(sd) == 5:
            open_t1 = float(sd.iloc[0]['开盘'])
            open_t5 = float(sd.iloc[4]['开盘'])
            close_t5 = float(sd.iloc[4]['收盘'])
            returns[code] = {
                'ret_open': (open_t5 - open_t1) / open_t1,
                'ret_close': (close_t5 - open_t1) / open_t1,
                'open_t1': open_t1,
                'open_t5': open_t5,
                'close_t5': close_t5,
            }
    return returns


def predict_at_date(processed, feature_cols, all_stocks, pred_date, model, config, device):
    """在指定日期预测所有股票的排序分数"""
    seq_len = config['sequence_length']
    sequences = []
    valid_stocks = []
    for code in all_stocks:
        hist = processed[
            (processed['股票代码'] == code) &
            (processed['日期'] <= pred_date)
        ].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            sequences.append(hist[feature_cols].values.astype(np.float32))
            valid_stocks.append(code)

    seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)
    with torch.no_grad():
        scores, aux = model(seq_tensor, return_aux=True)
        scores = scores.squeeze(0).cpu().numpy()
        predicted_returns = aux['return_abs'].squeeze(0).cpu().numpy()
    return scores, predicted_returns, valid_stocks


def compute_volatilities(raw_close, valid_stocks, pred_date):
    """计算每只股票的近期波动率（使用原始收盘价）"""
    vols = []
    for code in valid_stocks:
        sd = raw_close[
            (raw_close['股票代码'] == code) &
            (raw_close['日期'] <= pred_date)
        ].sort_values('日期')
        if len(sd) >= 20:
            closes = sd['收盘'].values.astype(float)
            rets = np.diff(closes) / (closes[:-1] + 1e-12)
            vols.append(np.std(rets[-20:]))
        else:
            vols.append(0.0)
    return np.array(vols)


def main():
    model, scaler, config, device = load_model()
    processed, feature_cols, all_stocks, train_df, test_df = prepare_features(config)

    # ⚠️ 保存原始收盘价用于波动率计算（必须在标准化之前）
    raw_close = processed[['股票代码', '日期', '收盘']].copy()

    # 标准化
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    processed[feature_cols] = processed[feature_cols].replace([np.inf, -np.inf], np.nan)
    processed[feature_cols] = processed[feature_cols].fillna(0)
    processed[feature_cols] = scaler.transform(processed[feature_cols])

    # 实际收益率
    actual_returns = compute_actual_returns(test_df)
    print(f"\nActual returns computed for {len(actual_returns)} stocks")

    # 在训练截止日预测
    pred_date = pd.to_datetime('2026-03-06')
    print(f"\nPredicting at: {pred_date.date()}")
    scores, predicted_returns, valid_stocks = predict_at_date(
        processed, feature_cols, all_stocks, pred_date, model, config, device
    )
    print(f"Valid stocks for prediction: {len(valid_stocks)}")

    volatilities = compute_volatilities(raw_close, valid_stocks, pred_date)

    # ========== Strategy 1: Top-5 Equal Weight ==========
    print("\n" + "=" * 70)
    print("Strategy 1: Top-5 by raw score (equal weight 0.2)")
    print("=" * 70)
    ranked = np.argsort(scores)[::-1]
    top5_idx = ranked[:5]
    score1 = 0.0
    for i, idx in enumerate(top5_idx):
        code = valid_stocks[idx]
        ret_info = actual_returns.get(code, {'ret_open': 0})
        ret = ret_info['ret_open'] if isinstance(ret_info, dict) else 0
        score1 += 0.2 * ret
        print(f"  #{i+1}: {code}  score={scores[idx]:.4f}  return={ret*100:+.2f}%")
    print(f"  => Score: {score1:.6f} ({score1*100:.4f}%)")

    # ========== Strategy 2: Optimized Weights (旧方法，保留对比) ==========
    print("\n" + "=" * 70)
    print("Strategy 2: Softmax + Volatility Penalty (old method, for reference)")
    print("=" * 70)
    old_top_indices, old_top_weights = optimize_weights(
        scores, volatilities=volatilities,
        top_k=5, candidate_k=10,
        use_volatility_penalty=True, temperature=2.0
    )
    score2 = 0.0
    for i, (idx, w) in enumerate(zip(old_top_indices, old_top_weights)):
        code = valid_stocks[idx]
        ret_info = actual_returns.get(code, {'ret_open': 0})
        ret = ret_info['ret_open'] if isinstance(ret_info, dict) else 0
        score2 += w * ret
        print(f"  #{i+1}: {code}  score={scores[idx]:.4f}  weight={w:.4f}  return={ret*100:+.2f}%")
    print(f"  => Score: {score2:.6f} ({score2*100:.4f}%)")

    # ========== Strategy 3: Return Gate (新方法) ==========
    print("\n" + "=" * 70)
    print("Strategy 3: Return Gate + Sharpe-like score (new method)")
    print("=" * 70)
    gate_indices, gate_weights = select_top_stocks_with_gate(
        scores, predicted_returns=predicted_returns,
        volatilities=volatilities,
        top_k=5, candidate_k=10,
        min_return_threshold=0.0,
        temperature=2.0, fallback='equal'
    )
    score3 = 0.0
    for i, (idx, w) in enumerate(zip(gate_indices, gate_weights)):
        code = valid_stocks[idx]
        ret_info = actual_returns.get(code, {'ret_open': 0})
        ret = ret_info['ret_open'] if isinstance(ret_info, dict) else 0
        score3 += w * ret
        print(f"  #{i+1}: {code}  score={scores[idx]:.4f}  pred_ret={predicted_returns[idx]*100:+.2f}%  "
              f"weight={w:.4f}  actual_ret={ret*100:+.2f}%")
    print(f"  => Score: {score3:.6f} ({score3*100:.4f}%)")

    # ========== Oracle ==========
    print("\n" + "=" * 70)
    print("Oracle: True Top-5 (best theoretically possible)")
    print("=" * 70)
    stock_returns_simple = {
        k: v['ret_open'] for k, v in actual_returns.items() if isinstance(v, dict)
    }
    true_ranked = sorted(stock_returns_simple.items(), key=lambda x: x[1], reverse=True)
    oracle_score = sum(r for _, r in true_ranked[:5]) / 5
    for i, (code, ret) in enumerate(true_ranked[:5]):
        print(f"  #{i+1}: {code}  return={ret*100:+.2f}%")
    print(f"  => Oracle Score: {oracle_score:.6f} ({oracle_score*100:.4f}%)")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Strategy 1 (equal weight):     {score1*100:.2f}%")
    print(f"  Strategy 2 (optimized weight): {score2*100:.2f}%")
    print(f"  Strategy 3 (return gate):      {score3*100:.2f}%")
    print(f"  Oracle (best possible):        {oracle_score*100:.2f}%")
    print(f"  ---")
    print(f"  1st place (7355608/Alka):      14.17%")
    print(f"  2nd place (O_O):               11.40%")
    print(f"  2nd place (Youzi):             11.40%")
    print(f"  ---")
    print(f"  Baseline return:               2.52%")
    print(f"  Your best return:              {max(score1, score2, score3)*100:.2f}%")
    print("=" * 70)

    # 保存自评结果
    result = {
        'prediction_date': str(pred_date.date()),
        'test_window': '2026-03-09 ~ 2026-03-13',
        'strategy1_equal_weight': float(score1),
        'strategy2_optimized_weight': float(score2),
        'strategy3_return_gate': float(score3),
        'oracle_max': float(oracle_score),
        'leaderboard_1st': 0.1417,
        'leaderboard_2nd': 0.1140,
        'baseline_return': 0.02518,
    }
    with open(os.path.join(MODEL_DIR, 'self_score.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSelf-score saved to: {MODEL_DIR}/self_score.json")


if __name__ == '__main__':
    main()
