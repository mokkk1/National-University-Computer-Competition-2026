"""
market_gate.py — 市场门控后处理模块 (V2 增强版)

改进:
  1. 三信号集成: model_head + RF分类器 + HS300动量
  2. 自适应防御权重: 根据信号置信度动态调整
  3. RF分类器使用全部18维宏观特征
  4. 支持 model_market_logits 作为额外信号源

用法:
  from market_gate import MarketGate, compute_market_signal, train_market_classifier

  signal = compute_market_signal(processed_df, pred_date, method='ensemble',
                                  macro_df=macro_df, rf_model=rf,
                                  model_market_logits=logits)
  gate = MarketGate(strategy='quarter_aware')
  top_indices, top_weights = gate.select(scores, stock_codes, market_signal, ...)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
import joblib

# 防御性行业
DEFENSIVE_INDUSTRIES = {
    '银行', '公用事业', '交通运输', '食品饮料', '医药生物',
    '建筑装饰', '建筑材料', '钢铁', '煤炭', '石油石化',
}

HIGH_BETA_INDUSTRIES = {
    '电子', '计算机', '传媒', '国防军工', '有色金属',
    '电力设备', '汽车', '机械设备', '非银金融',
}


def compute_hs300_return(processed_df, pred_date, lookback=10):
    """计算沪深300近N日累计涨跌幅"""
    df = processed_df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    mask = df['日期'] <= pred_date
    recent = df[mask].sort_values('日期')
    if recent.empty:
        return 0.0
    recent_dates = sorted(recent['日期'].unique())[-lookback:]
    if len(recent_dates) < 2:
        return 0.0
    daily_returns = []
    for d in recent_dates:
        day_data = recent[recent['日期'] == d]
        if '涨跌幅' in day_data.columns:
            avg_ret = day_data['涨跌幅'].mean()
            if not np.isnan(avg_ret):
                daily_returns.append(avg_ret / 100.0)
    if len(daily_returns) < 2:
        return 0.0
    cum_return = np.prod([1 + r for r in daily_returns]) - 1
    return float(cum_return)


def _sigmoid(x):
    """数值稳定的sigmoid"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_market_signal(processed_df, pred_date, method='hs300_return',
                          macro_df=None, rf_model=None,
                          model_market_logits=None):
    """
    综合市场信号计算 (V2: 支持多信号集成)

    Args:
        processed_df: 股票数据
        pred_date: 预测日期
        method: 'hs300_return' | 'rf_classifier' | 'ensemble' | 'model_head'
        macro_df: 宏观特征
        rf_model: RF分类器
        model_market_logits: 模型market_head的logits输出 (float or None)

    Returns:
        dict: {signal, direction, confidence, method, sub_signals}
    """
    sub_signals = {}

    # ─── 信号1: HS300动量 ───────────────────────
    hs300_ret = compute_hs300_return(processed_df, pred_date, lookback=10)
    hs300_direction = 1 if hs300_ret > 0.01 else (-1 if hs300_ret < -0.01 else 0)
    hs300_confidence = min(0.9, 0.5 + abs(hs300_ret) * 5)
    sub_signals['hs300'] = {'signal': hs300_ret, 'direction': hs300_direction,
                            'confidence': hs300_confidence}

    # ─── 信号2: 模型market_head ────────────────
    model_direction = 0
    model_confidence = 0.5
    model_prob = 0.5
    if model_market_logits is not None:
        model_prob = float(_sigmoid(model_market_logits))
        model_direction = 1 if model_prob > 0.55 else (-1 if model_prob < 0.45 else 0)
        model_confidence = max(model_prob, 1 - model_prob)
    sub_signals['model_head'] = {'signal': model_prob - 0.5, 'direction': model_direction,
                                  'confidence': model_confidence, 'prob_up': model_prob}

    # ─── 信号3: RF分类器 ───────────────────────
    rf_direction = 0
    rf_confidence = 0.5
    rf_prob = 0.5
    if rf_model is not None and method in ('rf_classifier', 'ensemble'):
        features = _build_market_features(processed_df, pred_date, macro_df)
        if features is not None and len(features) > 0:
            try:
                proba = rf_model.predict_proba(features.reshape(1, -1))[0]
                rf_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                rf_direction = 1 if rf_prob > 0.55 else (-1 if rf_prob < 0.45 else 0)
                rf_confidence = max(rf_prob, 1 - rf_prob)
            except Exception:
                pass
    sub_signals['rf'] = {'signal': rf_prob - 0.5, 'direction': rf_direction,
                         'confidence': rf_confidence, 'prob_up': rf_prob}

    # ─── 集成决策 ─────────────────────────────
    if method == 'ensemble':
        # 加权投票
        w_hs300 = 0.20
        w_model = 0.40
        w_rf = 0.40

        # 连续信号加权
        combined_signal = (
            w_hs300 * np.tanh(hs300_ret * 5) +       # HS300映射到[-1,1]
            w_model * (model_prob - 0.5) * 2 +         # model_head映射到[-1,1]
            w_rf * (rf_prob - 0.5) * 2                 # RF映射到[-1,1]
        )

        # 方向投票（多数决定）
        votes = []
        if hs300_direction != 0:
            votes.append((hs300_direction, hs300_confidence))
        if model_direction != 0:
            votes.append((model_direction, model_confidence))
        if rf_direction != 0:
            votes.append((rf_direction, rf_confidence))

        if votes:
            # 按置信度加权投票
            up_votes = sum(c for d, c in votes if d == 1)
            down_votes = sum(c for d, c in votes if d == -1)
            if up_votes > down_votes:
                direction = 1
            elif down_votes > up_votes:
                direction = -1
            else:
                direction = np.sign(combined_signal)
        else:
            direction = np.sign(combined_signal)

        confidence = min(0.95, 0.5 + abs(combined_signal) * 0.8)
        signal = combined_signal

    elif method == 'model_head' and model_market_logits is not None:
        signal = model_prob - 0.5
        direction = model_direction
        confidence = model_confidence

    elif method == 'rf_classifier' and rf_model is not None:
        signal = rf_prob - 0.5
        direction = rf_direction
        confidence = rf_confidence

    else:  # hs300_return (默认)
        signal = hs300_ret
        direction = hs300_direction
        confidence = hs300_confidence

    return {
        'signal': float(signal),
        'direction': int(direction),
        'confidence': float(confidence),
        'method': method,
        'sub_signals': sub_signals,
    }


# ─── V2: 增强版市场特征构建 ────────────────────────

def _build_market_features(processed_df, pred_date, macro_df=None):
    """构建市场方向分类器输入特征（使用全部18维宏观+市场宽度）"""
    df = processed_df[processed_df['日期'] <= pred_date].copy()
    if df.empty:
        return None

    features = []
    dates_sorted = sorted(df['日期'].unique())

    # 1. 市场收益特征 (6维)
    for lb in [3, 5, 10, 20]:
        recent_dates = dates_sorted[-lb:]
        if '涨跌幅' in df.columns:
            rets = []
            for d in recent_dates:
                day_ret = df[df['日期'] == d]['涨跌幅'].mean()
                if not np.isnan(day_ret):
                    rets.append(day_ret / 100)
            if rets:
                features.append(np.mean(rets))
                features.append(np.std(rets) if len(rets) > 1 else 0.0)
                features.append(np.min(rets))
                features.append(np.max(rets))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            if lb == 10:  # 只保留10日的详细特征
                pass
            else:  # 简化
                features = features[:-2]  # 去掉min/max，只保留mean/std
        else:
            features.extend([0.0, 0.0])

    # 2. 涨跌家数比 (1维)
    last_date = dates_sorted[-1]
    last_day = df[df['日期'] == last_date]
    if '涨跌幅' in last_day.columns:
        up_ratio = (last_day['涨跌幅'] > 0).mean()
        down_ratio = (last_day['涨跌幅'] < 0).mean()
        features.append(up_ratio if not np.isnan(up_ratio) else 0.5)
        features.append(down_ratio if not np.isnan(down_ratio) else 0.5)
    else:
        features.extend([0.5, 0.5])

    # 3. 收益离散度和偏度 (2维)
    if '涨跌幅' in last_day.columns:
        rets = last_day['涨跌幅'].dropna() / 100
        if len(rets) > 0:
            features.append(float(rets.std()))
            features.append(float(rets.skew()) if len(rets) > 2 else 0.0)
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    # 4. 成交量特征 (2维)
    if '成交量' in last_day.columns:
        recent_vol = df.groupby('日期')['成交量'].mean()
        if len(recent_vol) >= 5:
            vol_ma5 = recent_vol.iloc[-5:].mean()
            vol_ma20 = recent_vol.iloc[-20:].mean() if len(recent_vol) >= 20 else vol_ma5
            features.append(float(recent_vol.iloc[-1] / (vol_ma5 + 1) - 1))
            features.append(float(vol_ma5 / (vol_ma20 + 1) - 1))
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    # 5. 宏观特征 (18维) — 取最新值和变化
    if macro_df is not None:
        macro_hist = macro_df[macro_df['日期'] <= pred_date].sort_values('日期')
        if not macro_hist.empty:
            latest = macro_hist.iloc[-1]
            macro_cols = [c for c in macro_df.columns if c != '日期']
            for col in macro_cols:
                if col in latest.index:
                    val = latest[col]
                    features.append(float(val) if not np.isnan(val) else 0.0)

            # 宏观特征5日变化（如果有足够历史）
            if len(macro_hist) >= 5:
                prev = macro_hist.iloc[-5]
                for col in macro_cols:
                    if col in latest.index and col in prev.index:
                        diff = float(latest[col]) - float(prev[col])
                        if not np.isnan(diff):
                            features.append(diff)
                        else:
                            features.append(0.0)
                    else:
                        features.append(0.0)

    return np.array(features, dtype=np.float32)


def train_market_classifier(processed_df, macro_df=None, data_dir=None):
    """
    训练增强版随机森林市场方向分类器 (V2: 使用全部宏观特征)

    Returns:
        rf_model, accuracy, feature_importance
    """
    df = processed_df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    dates = sorted(df['日期'].unique())

    X_list, y_list = [], []

    print(f"  构建训练样本 (日期范围: {dates[30]} ~ {dates[-6]})...")
    for i, pred_date in enumerate(dates[30:-5]):
        features = _build_market_features(df, pred_date, macro_df)
        if features is None:
            continue

        # 标签：未来5日市场涨跌
        future_dates = [d for d in dates if d > pred_date][:5]
        if len(future_dates) < 5:
            continue

        future_ret = 0.0
        for fd in future_dates:
            day_df = df[df['日期'] == fd]
            if '涨跌幅' in day_df.columns:
                future_ret += day_df['涨跌幅'].mean() / 100

        X_list.append(features)
        y_list.append(1 if future_ret > 0 else 0)

    if len(X_list) < 20:
        print("⚠️ 训练样本不足，无法训练市场分类器")
        return None, 0.0, None

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"  样本数: {len(X)}, 特征维度: {X.shape[1]}, 正例比例: {y.mean():.1%}")

    # 时间序列分割（后20%作为验证）
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=5, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        rf.fit(X_train, y_train)

        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test) if len(X_test) > 0 else train_acc

    # 特征重要性
    feature_importance = None
    if hasattr(rf, 'feature_importances_'):
        feature_importance = rf.feature_importances_

    print(f"✅ RF市场分类器: 训练准确率={train_acc:.2%}, 验证准确率={test_acc:.2%}")

    # 保存模型
    if data_dir:
        os.makedirs(os.path.join(data_dir, 'model'), exist_ok=True)
        model_path = os.path.join(data_dir, 'model', 'market_rf_classifier.pkl')
        joblib.dump(rf, model_path)
        print(f"  模型已保存: {model_path}")

    return rf, test_acc, feature_importance


# ─── 防御性评分 ────────────────────────────────────

def compute_stock_defensive_score(stock_codes, processed_df, pred_date):
    """计算每只股票的防御性评分"""
    df = processed_df[processed_df['日期'] <= pred_date]
    N = len(stock_codes)
    scores = np.zeros(N)

    for i, code in enumerate(stock_codes):
        stock_data = df[df['股票代码'] == code].sort_values('日期')
        if len(stock_data) < 20:
            scores[i] = 0.5
            continue

        score = 0.0

        # 低波动 (35%)
        if '涨跌幅' in stock_data.columns:
            returns = stock_data['涨跌幅'].tail(20) / 100
            vol = returns.std()
            score += (1.0 - min(vol / 0.05, 1.0)) * 0.35

        # 大市值 (25%)
        if '成交额' in stock_data.columns:
            avg_amount = stock_data['成交额'].tail(20).mean()
            log_amount = np.log1p(avg_amount)
            amount_score = min(log_amount / 20, 1.0)
            score += amount_score * 0.25

        # 低beta (25%)
        if '涨跌幅' in stock_data.columns:
            stock_rets = stock_data['涨跌幅'].tail(60) / 100
            market_rets = []
            all_dates = stock_data['日期'].tail(60)
            for d in all_dates:
                day_df = df[df['日期'] == d]
                if '涨跌幅' in day_df.columns:
                    market_rets.append(day_df['涨跌幅'].mean() / 100)
            if len(stock_rets) >= 10 and len(market_rets) >= 10:
                min_len = min(len(stock_rets), len(market_rets))
                beta = np.corrcoef(stock_rets.iloc[-min_len:], market_rets[-min_len:])[0, 1]
                if not np.isnan(beta):
                    score += (1.0 - min(max(beta, 0), 1.5) / 1.5) * 0.25
                else:
                    score += 0.125
            else:
                score += 0.125

        # 行业防御性 (15%)
        if 'industry' in stock_data.columns:
            industry_val = stock_data['industry'].iloc[-1]
            # 整数编码的行业，暂不加分
            pass

        scores[i] = score

    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores


# ─── MarketGate (V2: 自适应防御权重) ──────────────

class MarketGate:
    """市场门控后处理器 (V2: 自适应防御权重 + 信号集成)"""

    def __init__(self, strategy='adaptive', defensive_weight=0.6):
        self.strategy = strategy
        self.defensive_weight = defensive_weight

    def _compute_days_to_qe(self, pred_date):
        m, y = pred_date.month, pred_date.year
        if m <= 3:
            qe = pd.Timestamp(year=y, month=3, day=31)
        elif m <= 6:
            qe = pd.Timestamp(year=y, month=6, day=30)
        elif m <= 9:
            qe = pd.Timestamp(year=y, month=9, day=30)
        else:
            qe = pd.Timestamp(year=y, month=12, day=31)
        return max(0.0, (qe - pred_date).days) / 90.0

    def _adaptive_defensive_weight(self, direction, confidence, days_to_qe):
        """
        V2: 自适应防御权重。

        逻辑:
        - 基础防御权重 = 0.3（总是保留一定防御性）
        - 看跌信号: 防御权重 = 0.3 + 0.6 * confidence（范围 0.3~0.9）
        - 季末日额外加成: +0.15（季末不确定性）
        - 看涨信号: 防御权重 = 0.1（几乎不防御）
        """
        if direction < 0:
            # 看跌：防御权重随置信度线性增长
            base = 0.3 + 0.5 * confidence
            # 季末加成
            if days_to_qe < 0.055:  # 5天内季末
                base = min(0.95, base + 0.2)
            return min(0.95, base)
        elif direction == 0:
            # 中性：中等防御
            neutral = 0.3
            if days_to_qe < 0.055:
                neutral = 0.45
            return neutral
        else:
            # 看涨：轻度防御
            if days_to_qe < 0.055:
                return 0.35  # 季末看涨仍然谨慎
            return 0.15

    def select(self, scores, stock_codes, market_signal,
               predicted_returns=None, volatilities=None,
               processed_df=None, pred_date=None,
               top_k=5, candidate_k=10, temperature=2.0):
        """
        V2: 自适应防御权重选股。

        策略:
          - 'quarter_aware': 季末感知 + 自适应权重 (V2默认)
          - 'adaptive': 纯信号驱动 + 自适应权重
          - 'always_normal' / 'always_defensive': 固定策略
        """
        direction = market_signal.get('direction', 0)
        confidence = market_signal.get('confidence', 0.5)
        signal = market_signal.get('signal', 0.0)

        if pred_date is None:
            pred_date = pd.Timestamp.now()

        days_to_qe = self._compute_days_to_qe(pred_date)

        if self.strategy == 'always_normal':
            return self._select_normal(scores, predicted_returns, volatilities,
                                       top_k, candidate_k, temperature)
        elif self.strategy == 'always_defensive':
            return self._select_defensive(scores, stock_codes, predicted_returns,
                                          volatilities, processed_df, pred_date,
                                          top_k, candidate_k, temperature,
                                          override_weight=0.95)

        # ─── V2 主逻辑: 自适应权重 ───
        is_quarter_end = days_to_qe < 0.055

        if self.strategy in ('quarter_aware', 'adaptive'):
            defensive_weight = self._adaptive_defensive_weight(direction, confidence, days_to_qe)

            # 打印策略信息
            if is_quarter_end and direction < 0:
                tag = f"🏃 季末+看跌 → 强防御 w={defensive_weight:.2f}"
            elif is_quarter_end and direction >= 0:
                tag = f"⚠️ 季末+看涨 → 谨慎防御 w={defensive_weight:.2f}"
            elif direction < 0:
                tag = f"🛡️ 看跌 → 防御 w={defensive_weight:.2f}"
            elif direction > 0:
                tag = f"🚀 看涨 → 轻防御 w={defensive_weight:.2f}"
            else:
                tag = f"📊 中性 → 平衡 w={defensive_weight:.2f}"

            # 显示子信号
            sub = market_signal.get('sub_signals', {})
            if sub:
                parts = []
                for name, s in sub.items():
                    d_emoji = {1: '↑', -1: '↓', 0: '→'}.get(s.get('direction', 0), '?')
                    parts.append(f"{name}={d_emoji}")
                tag += f" ({' '.join(parts)})"

            print(f"    {tag}")

            return self._select_defensive(
                scores, stock_codes, predicted_returns,
                volatilities, processed_df, pred_date,
                top_k, candidate_k, temperature,
                override_weight=defensive_weight
            )

        else:
            # 回退到简单逻辑
            if direction >= 0:
                return self._select_normal(scores, predicted_returns, volatilities,
                                           top_k, candidate_k, temperature)
            else:
                return self._select_defensive(scores, stock_codes, predicted_returns,
                                              volatilities, processed_df, pred_date,
                                              top_k, candidate_k, temperature,
                                              override_weight=self.defensive_weight)

    def _select_normal(self, scores, predicted_returns, volatilities,
                       top_k, candidate_k, temperature):
        from utils import select_top_stocks_with_gate
        return select_top_stocks_with_gate(
            scores, predicted_returns=predicted_returns,
            volatilities=volatilities, top_k=top_k,
            candidate_k=candidate_k, min_return_threshold=0.0,
            temperature=temperature, fallback='equal'
        )

    def _select_defensive(self, scores, stock_codes, predicted_returns,
                           volatilities, processed_df, pred_date,
                           top_k, candidate_k, temperature,
                           override_weight=None):
        dw = override_weight if override_weight is not None else self.defensive_weight
        N = len(scores)
        k_candidate = min(candidate_k * 2, N)

        if processed_df is not None and pred_date is not None:
            defensive_scores = compute_stock_defensive_score(stock_codes, processed_df, pred_date)
        else:
            defensive_scores = np.ones(N) * 0.5

        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        combined = (1 - dw) * scores_norm + dw * defensive_scores

        if predicted_returns is not None:
            return_factor = 1.0 + predicted_returns
            return_factor = np.clip(return_factor, 0.1, 5.0)
            combined = combined * return_factor

        if volatilities is not None:
            vol_penalty = 1.0 + np.abs(volatilities)
            combined = combined / (vol_penalty + 1e-12)

        candidate_indices = np.argsort(combined)[::-1][:k_candidate]
        candidate_combined = combined[candidate_indices]

        scaled = candidate_combined / temperature
        scaled = scaled - np.max(scaled)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / exp_scaled.sum()

        final_order = np.argsort(weights)[::-1]
        top_indices = candidate_indices[final_order[:top_k]]
        top_weights = weights[final_order[:top_k]]
        top_weights = top_weights / top_weights.sum()

        return top_indices, top_weights
