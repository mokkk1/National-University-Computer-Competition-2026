"""
self_eval_ensemble.py — 多 seed 集成自评

与 self_eval.py 的区别：
  - 同一窗口的多个 seed 模型做 median 分数集成
  - 显示各股票的 seed 共识度
  - 支持 --seeds 42,123,456

用法：
  python self_eval_ensemble.py --config standard --seeds 42,123,456
"""

import pandas as pd, numpy as np, torch, joblib, os, sys, json, argparse
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))
from model import StockTransformer
from utils import engineer_features_158plus39
from train import feature_cloums_map, feature_engineer_func_map
from walk_forward import load_walk_forward_models

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BASE = os.path.join(PROJECT_DIR, 'model', 'walk_forward')

WINDOW_DATES = {
    'window_01': '2024-09-30', 'window_02': '2024-12-31',
    'window_03': '2025-03-31', 'window_04': '2025-06-30',
    'window_05': '2025-09-30', 'window_06': '2025-12-31',
}


def load_data():
    train = pd.read_csv(os.path.join(PROJECT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(PROJECT_DIR, 'test.csv'))
    full = pd.concat([train, test], ignore_index=True)
    full['日期'] = pd.to_datetime(full['日期'])
    return full.sort_values(['股票代码', '日期']).reset_index(drop=True)


def build_features(full_df, feature_num='158+39+fundamental+momentum'):
    feature_engineer = feature_engineer_func_map.get(feature_num, engineer_features_158plus39)
    print(f"  特征工程 ({feature_num})...")
    groups = [g for _, g in full_df.groupby('股票代码', sort=False)]
    processed_list = [feature_engineer(g) for g in tqdm(groups, desc="  特征工程")]
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # 宏观 + 行业
    try:
        from macro_industry import load_macro_features, merge_macro_to_stock, add_industry_features
        macro_df = load_macro_features()
        if macro_df is not None:
            processed = merge_macro_to_stock(processed, macro_df)
    except Exception:
        pass
    try:
        from macro_industry import add_industry_features
        processed, _ = add_industry_features(processed)
    except Exception:
        pass
    if 'industry' not in processed.columns:
        processed['industry'] = 0

    all_stocks = sorted(processed['股票代码'].unique())
    stock2idx = {s: i for i, s in enumerate(all_stocks)}
    processed['instrument'] = processed['股票代码'].map(stock2idx)

    fc_template = feature_cloums_map.get(feature_num, feature_cloums_map['158+39'])
    feature_cols = [c for c in fc_template if c in processed.columns]
    for mc in processed.columns:
        if mc.startswith(('bond', 'north', 'margin', 'usdcny', 'lpr', 'shibor', 'cpi', 'pmi', 'm1', 'm2', 'social')) and mc not in feature_cols:
            feature_cols.append(mc)
    if 'industry' in processed.columns and 'industry' not in feature_cols:
        feature_cols.append('industry')
    return processed, feature_cols, stock2idx


def predict_at_date(model_info, processed_df, feature_cols, stock_codes, pred_date, device):
    cfg = model_info['config']
    seq_len = cfg.get('sequence_length', 60)
    sequences, valid_stocks = [], []
    for code in stock_codes:
        hist = processed_df[(processed_df['股票代码'] == code) & (processed_df['日期'] <= pred_date)].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            sequences.append(hist[feature_cols].values.astype(np.float32))
            valid_stocks.append(code)
    if not sequences:
        return None
    seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model_info['model'](seq_tensor)
    return {'scores': scores.squeeze(0).cpu().numpy(), 'stocks': valid_stocks}


def compute_future_returns(full_df, pred_date, horizon=5):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='standard')
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--wf-dir', type=str, default=MODEL_BASE)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}, seeds: {seeds}")

    # 加载模型
    models = load_walk_forward_models(args.wf_dir, args.config, seeds, device)
    print(f"加载了 {len(models)} 个模型")

    if len(models) == 0:
        print("无模型！")
        sys.exit(1)

    # 按窗口分组
    window_models = defaultdict(list)
    for m in models:
        win = os.path.basename(m['window_dir'])
        window_models[win].append(m)

    print(f"窗口数: {len(window_models)}, 每窗口模型数: {[len(v) for v in window_models.values()]}")

    # 数据
    print("\n加载数据...")
    full_df = load_data()
    first_cfg = models[0]['config']
    processed_df, feature_cols, stock2idx = build_features(full_df, first_cfg.get('feature_num', '158+39+fundamental+momentum'))
    stock_codes = sorted(processed_df['股票代码'].unique())

    # 评估
    results = []
    print(f"\n{'='*70}")
    print("Walk-Forward 多Seed集成回测")
    print(f"{'='*70}")

    for win_idx, (window_dir, win_models) in enumerate(sorted(window_models.items())):
        pred_date_str = WINDOW_DATES.get(window_dir)
        if not pred_date_str:
            continue
        pred_date = pd.to_datetime(pred_date_str)

        n_seeds = len(win_models)
        print(f"\nW{win_idx+1} ({pred_date_str}): {n_seeds} seeds")

        all_scores, all_stocks, consensus = [], None, Counter()

        for m in win_models:
            cfg, scaler = m['config'], m['scaler']
            sf = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else [c for c in feature_cols if c != 'industry']
            edf = processed_df.copy()
            for c in sf:
                if c not in edf.columns:
                    edf[c] = 0.0
            if cfg.get('use_industry_embedding') and 'industry' not in edf.columns:
                edf['industry'] = 0
            sc = [c for c in sf if c in edf.columns]
            edf[sc] = edf[sc].replace([np.inf, -np.inf], np.nan).fillna(0)
            edf[sc] = scaler.transform(edf[sc])
            mc = sc + (['industry'] if cfg.get('use_industry_embedding') else [])

            r = predict_at_date(m, edf, mc, stock_codes, pred_date, device)
            if r is None:
                continue
            if all_stocks is None:
                all_stocks = r['stocks']
            s2s = dict(zip(r['stocks'], r['scores']))
            all_scores.append(np.array([s2s.get(s, -1e9) for s in all_stocks]))
            for idx in np.argsort(r['scores'])[-5:][::-1]:
                consensus[r['stocks'][idx]] += 1

        if not all_scores:
            continue

        median_scores = np.median(all_scores, axis=0)
        ti = np.argsort(median_scores)[-5:][::-1]

        print(f"  Top-5:")
        for rank, idx in enumerate(ti):
            code = all_stocks[idx]
            print(f"    #{rank+1}: {code}  score={median_scores[idx]:.4f}  "
                  f"consensus={consensus.get(code,0)}/{n_seeds}")

        actual = compute_future_returns(full_df, pred_date)
        weights = np.exp(median_scores[ti]) / np.exp(median_scores[ti]).sum()
        pret = sum(w * actual.get(all_stocks[i], 0) for i, w in zip(ti, weights))

        o_stocks = sorted(actual.items(), key=lambda x: x[1], reverse=True)[:5]
        o_ret = sum(r for _, r in o_stocks) / 5 if o_stocks else 0

        marker = "✅" if pret > 0 else "❌"
        print(f"  {marker} 组合收益={pret*100:+.2f}%  Oracle={o_ret*100:+.2f}%")

        results.append({'window': window_dir, 'pred_date': pred_date_str,
                        'portfolio_return': pret, 'oracle_return': o_ret,
                        'n_seeds': n_seeds, 'top_stocks': [all_stocks[i] for i in ti]})

    # 摘要
    if results:
        rets = [r['portfolio_return'] for r in results]
        print(f"\n{'='*70}")
        print("摘要")
        print(f"{'='*70}")
        for r in results:
            m = "✅" if r['portfolio_return'] > 0 else "❌"
            print(f"  {m} {r['pred_date']}: {r['portfolio_return']*100:+.2f}% ({r['n_seeds']} seeds)")
        print(f"\n  均值: {np.mean(rets)*100:+.2f}%  中位数: {np.median(rets)*100:+.2f}%  "
              f"正收益: {sum(1 for r in rets if r > 0)}/{len(rets)}")
        print(f"  V7官方: -1.29%  基线: -2.15%")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.wf_dir, f'self_eval_ensemble_{ts}.json')
    with open(out, 'w') as f:
        json.dump({'results': [{k: float(v) if isinstance(v, (np.floating,)) else v for k, v in r.items()} for r in results]}, f, indent=2)
    print(f"\n保存: {out}")


if __name__ == '__main__':
    main()
