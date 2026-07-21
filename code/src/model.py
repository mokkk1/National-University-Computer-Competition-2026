"""
StockTransformer — 基于 Transformer 的沪深300股票排序模型。

本模块实现了完整的排序学习神经网络架构，包含以下核心组件：

1. **时序编码** — MultiScaleConv (多尺度卷积) + TransformerEncoder (3层自注意力)
2. **特征交互** — FeatureInteraction (低秩 bilinear 交叉) + FeatureAttention (时序聚合)
3. **股票间交互** — CrossStockAttention (同日股票间多头注意力)
4. **市场聚合** — MarketAttentionPooling (注意力池化→市场状态向量) + MarketGate (门控调制)
5. **排序头** — 多层 MLP 输出排序分数 + 辅助任务头 (方向/波动/收益)
6. **轻量替代** — LightweightStockRanker (~264K 参数, 11×削减)

主要入口:
    StockTransformer(input_dim, config, num_stocks)  — 完整架构 (~2.5M 参数)
    LightweightStockRanker(input_dim, config)        — 轻量架构 (~264K 参数)

参考:
    - 竞赛: THU-BDC2026 沪深300指数预测
    - 架构设计: 改进方案.md §1.1 (市场聚合), §2.2 (Portfolio Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 多尺度时序卷积模块
class MultiScaleConv(nn.Module):
    """
    多尺度时序卷积，在 Transformer 之前提取不同周期模式。

    使用多个不同 kernel size 的 1D 卷积并行处理时序：
    - kernel=3: 捕捉短期动量/反转
    - kernel=5: 捕捉中期趋势变化
    - kernel=7: 捕捉长周期模式

    各尺度输出拼接后通过残差连接回到原维度。
    """
    def __init__(self, d_model, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(MultiScaleConv, self).__init__()
        self.kernel_sizes = kernel_sizes
        # 每个 kernel 输出 d_model // len(kernel_sizes) 维
        out_dim_per_kernel = d_model // len(kernel_sizes)
        remainder = d_model - out_dim_per_kernel * len(kernel_sizes)
        self.out_dims = [out_dim_per_kernel + (1 if i < remainder else 0)
                         for i in range(len(kernel_sizes))]

        self.convs = nn.ModuleList()
        for k, od in zip(kernel_sizes, self.out_dims):
            padding = (k - 1) // 2  # same padding
            self.convs.append(nn.Sequential(
                nn.Conv1d(d_model, od, kernel_size=k, padding=padding),
                nn.BatchNorm1d(od),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        # 输出投影 + 残差
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        # Conv1d 需要 [B, C, L]
        x_t = x.transpose(1, 2)  # [B*N, d_model, seq_len]

        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x_t))  # [B*N, od, seq_len]

        # 拼接各尺度输出
        x_cat = torch.cat(conv_outputs, dim=1)  # [B*N, d_model, seq_len]
        x_cat = x_cat.transpose(1, 2)  # [B*N, seq_len, d_model]

        # 残差连接
        output = self.norm(x + self.output_proj(x_cat))
        return output


# 特征交互模块
class FeatureInteraction(nn.Module):
    """
    特征交互层，让模型学习"量价配合"、"趋势与波动结合"等组合信号。

    使用低秩 bilinear 交互（类似 DCN 的 cross network 简化版）：
    - 对每对特征维度计算交互，低秩近似控制参数量
    - 交互结果与原始特征融合
    """
    def __init__(self, d_model, rank=64, dropout=0.1):
        super(FeatureInteraction, self).__init__()
        self.rank = rank
        # 双向低秩投影
        self.W_a = nn.Linear(d_model, rank, bias=False)
        self.W_b = nn.Linear(d_model, rank, bias=False)
        # 交互结果投影回原空间
        self.output_proj = nn.Linear(rank, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch*num_stocks, d_model]
        # 低秩特征交叉
        a = self.W_a(x)  # [..., rank]
        b = self.W_b(x)  # [..., rank]
        # Hadamard 积实现特征交互
        interaction = a * b  # [..., rank]
        interaction = self.output_proj(interaction)  # [..., d_model]
        # 残差融合
        output = self.norm(x + self.dropout(interaction))
        return output


# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：每只股票都关注其他股票的特征
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        output = self.norm(stock_features + self.dropout(attended))
        return output


class MarketAttentionPooling(nn.Module):
    """
    市场注意力池化层 (★ 架构级改进)

    将 Cross-Stock Attention 输出的 300 个股票特征聚合为一个市场状态向量。
    使用注意力池化（非简单均值），让模型学习哪些股票对市场状态判断最重要。

    直觉：银行股权重高 → 市场由金融驱动；科技股权重高 → 市场由成长驱动
    """
    def __init__(self, d_model, market_dim=64, nhead=4, dropout=0.1):
        super(MarketAttentionPooling, self).__init__()
        self.market_dim = market_dim
        # 多头注意力池化：学习多个市场视角
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # 可学习的 query token（代表"市场状态"这个抽象概念）
        self.market_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # 投影到 market_dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, market_dim),
            nn.LayerNorm(market_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, stock_features, masks=None):
        """
        Args:
            stock_features: [B, N, d_model] Cross-Stock Attention 输出
            masks: [B, N] or None, 有效股票掩码

        Returns:
            market_embedding: [B, market_dim] 市场状态向量
        """
        B, N, D = stock_features.shape

        # 扩展 query token 到 batch
        query = self.market_query.expand(B, -1, -1)  # [B, 1, d_model]

        # 处理 mask: MultiheadAttention 需要 key_padding_mask
        key_padding_mask = None
        if masks is not None:
            # masks: [B, N]  True=padding, False=valid → 需要 ~mask
            key_padding_mask = (masks < 0.5)  # [B, N]

        # 注意力池化：query 关注所有股票
        pooled, attn_weights = self.attention(
            query, stock_features, stock_features,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )  # pooled: [B, 1, d_model]

        # 投影到 market_dim
        market_embedding = self.proj(pooled.squeeze(1))  # [B, market_dim]

        return market_embedding, attn_weights


class MarketGate(nn.Module):
    """
    市场门控层 (★ 架构级改进)

    将市场状态向量拼回个股特征，通过门控机制调制排序分数。

    核心思想：
    - 市场预测下跌时 → 防御性股票（低波动、大市值）的分数自动上升
    - 市场预测上涨时 → 进攻性股票（高动量、高beta）的分数自动上升

    实现：
      gate = sigmoid(W_g * [stock_i, market_embedding])
      modulated_i = stock_i * gate + market_bias
    """
    def __init__(self, d_model, market_dim=64, dropout=0.1):
        super(MarketGate, self).__init__()
        input_dim = d_model + market_dim
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # gate ∈ (0, 1)
        )
        # 偏置网络（market → per-stock adjustment）
        self.bias_net = nn.Sequential(
            nn.Linear(market_dim, d_model),
            nn.Tanh()  # bias ∈ (-1, 1)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features, market_embedding):
        """
        Args:
            stock_features: [B*N, d_model] 展平后的个股特征
            market_embedding: [B, market_dim] 市场状态向量

        Returns:
            modulated_features: [B*N, d_model] 市场调制后的个股特征
        """
        B_mul_N, D = stock_features.shape
        B = market_embedding.shape[0]
        N = B_mul_N // B

        # 将 market_embedding 扩展到每只股票
        market_expanded = market_embedding.unsqueeze(1).expand(B, N, -1)  # [B, N, market_dim]
        market_flat = market_expanded.reshape(B * N, -1)  # [B*N, market_dim]

        # 拼接个股特征与市场状态
        combined = torch.cat([stock_features, market_flat], dim=-1)  # [B*N, D+market_dim]

        # 门控调制
        gate = self.gate_net(combined)  # [B*N, d_model]
        bias = self.bias_net(market_flat)  # [B*N, d_model]

        modulated = stock_features * gate + bias
        modulated = self.norm(modulated + stock_features)  # 残差连接

        return self.dropout(modulated)

class FeatureAttention(nn.Module):
    """
    特征注意力模块 — 学习时序中哪些时间步对排序最重要.

    对每个时间步计算注意力权重 (通过 MLP → Softmax), 加权求和压缩时序维度.
    输出: [batch*num_stocks, d_model] 的聚合特征向量.
    """
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_weights = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)

class StockTransformer(nn.Module):
    """
    基于 Transformer 的沪深300股票排序模型 (~2.5M 参数)。

    数据流::

        [B, 300, 60, 237] 股票特征
            │
            ├── Input Projection (237 → d_model)
            ├── Positional Encoding
            ├── MultiScaleConv (kernel=3/5/7 多尺度时序卷积)
            ├── TransformerEncoder (3层, 4头自注意力)
            ├── FeatureAttention (学习哪些时间步重要)
            ├── FeatureInteraction (特征间低秩交叉)
            │
            ├── CrossStockAttention (同日股票间多头注意力)
            │
            ├── [可选] MarketAttentionPooling → MarketGate
            │     ├── market_head → BCE(涨/跌)
            │     └── 门控调制个股排序分数
            │
            ├── Ranking Layers (d_model → d_model//2)
            ├── Score Head → 排序分数 [B, 300]
            ├── Direction Head → 涨跌方向 (辅助)
            ├── Volatility Head → 波动率预测 (辅助)
            └── Return Head → 绝对收益 (Huber回归, 辅助)

    Parameters
    ----------
    input_dim : int
        输入特征维度 (不含 industry ID, 如 237).
    config : dict
        训练配置字典, 包含 d_model, nhead, num_layers, dropout 等.
    num_stocks : int
        股票数量 (沪深300 = 300).
    emb_dim : int
        行业 embedding 维度, 默认 16.

    Input Shape
    -----------
    src : [batch, num_stocks, seq_len, input_dim+1]
        最后一列为 industry ID (若 use_industry_embedding=True).

    Output Shape
    -----------
    scores : [batch, num_stocks]
        每只股票的排序分数, 越高越好.
    aux : dict (仅 return_aux=True)
        'direction', 'volatility', 'return_abs': [batch*num_stocks]
        'market_logits': [batch] (若 use_market_aggregation=True)
    """
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks

        # ─── 行业 Embedding ──────────────────────────
        self.use_industry_embedding = config.get('use_industry_embedding', False)
        if self.use_industry_embedding:
            num_industries = config.get('num_industries', 31)
            industry_emb_dim = config.get('industry_emb_dim', 16)
            self.industry_embedding = nn.Embedding(num_industries, industry_emb_dim)
            effective_d_model = config['d_model'] + industry_emb_dim
        else:
            effective_d_model = config['d_model']

        # ─── 市场聚合架构 (★ P1 核心改进) ──────────
        self.use_market_aggregation = config.get('use_market_aggregation', False)
        self.market_dim = config.get('market_dim', 64)
        if self.use_market_aggregation:
            self.market_pooling = MarketAttentionPooling(
                effective_d_model,
                market_dim=self.market_dim,
                nhead=config.get('market_pool_heads', 4),
                dropout=config['dropout']
            )
            self.market_gate = MarketGate(
                effective_d_model,
                market_dim=self.market_dim,
                dropout=config['dropout']
            )
            # 市场方向预测头：涨/跌 二分类
            self.market_head = nn.Sequential(
                nn.Linear(self.market_dim, self.market_dim // 2),
                nn.LayerNorm(self.market_dim // 2),
                nn.GELU(),
                nn.Dropout(config['dropout'] * 0.5),
                nn.Linear(self.market_dim // 2, 1)  # 输出 logit
            )

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])

        # 多尺度时序卷积（可选）
        self.use_tcn = config.get('use_tcn', True)
        if self.use_tcn:
            self.multi_scale_conv = MultiScaleConv(
                config['d_model'],
                kernel_sizes=config.get('tcn_kernel_sizes', [3, 5, 7]),
                dropout=config.get('tcn_dropout', 0.1)
            )

        # 时序特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 特征注意力
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])

        # 特征交互（可选）
        self.use_feature_interaction = config.get('use_feature_interaction', True)
        if self.use_feature_interaction:
            self.feature_interaction = FeatureInteraction(
                config['d_model'],
                rank=config.get('fi_rank', 64),
                dropout=config['dropout']
            )

        # 股票间交互注意力（使用行业增强后的维度）
        self.cross_stock_attention = CrossStockAttention(effective_d_model, config['nhead'], config['dropout'])

        # 排序特异性层
        self.ranking_layers = nn.Sequential(
            nn.Linear(effective_d_model, effective_d_model),
            nn.LayerNorm(effective_d_model),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(effective_d_model, effective_d_model // 2),
            nn.LayerNorm(effective_d_model // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        self._ranking_dim = effective_d_model // 2

        # 最终排序分数输出
        half_dim = effective_d_model // 2
        self.score_head = nn.Sequential(
            nn.Linear(half_dim, half_dim // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(half_dim // 2, 1)
        )
        # 辅助任务头
        self.direction_head = nn.Sequential(
            nn.Linear(half_dim, half_dim // 2), nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5), nn.Linear(half_dim // 2, 1)
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(half_dim, half_dim // 2), nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5), nn.Linear(half_dim // 2, 1)
        )
        self.return_head = nn.Sequential(
            nn.Linear(half_dim, half_dim // 2), nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5), nn.Linear(half_dim // 2, 1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, return_aux=False, stock_masks=None):
        """
        Args:
            src: [batch, num_stocks, seq_len, feature_dim]
                 如果 use_industry_embedding=True，最后一列为 industry 整数编码
            return_aux: 是否返回辅助任务输出
            stock_masks: [batch, num_stocks] 有效股票掩码（可选）

        Returns:
            scores: [batch, num_stocks]
            aux: (仅 return_aux=True)
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # ─── 提取行业 ID ──────────────────────────
        if self.use_industry_embedding:
            # 最后一列是 industry ID，取第一个时间步的值（所有时间步相同）
            industry_ids = src[:, :, 0, -1].long()  # [B, N]
            # 去掉最后一列
            src = src[:, :, :, :-1]
            feature_dim -= 1

        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.contiguous().view(batch_size * num_stocks, seq_len, feature_dim)

        # 输入投影和位置编码
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        # 多尺度时序卷积
        if self.use_tcn:
            src_proj = self.multi_scale_conv(src_proj)

        # 时序特征提取
        temporal_features = self.temporal_encoder(src_proj)

        # 特征注意力聚合
        aggregated_features = self.feature_attention(temporal_features)

        # 特征交互
        if self.use_feature_interaction:
            aggregated_features = self.feature_interaction(aggregated_features)

        # 重塑回股票维度
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)

        # ─── 注入行业 Embedding ─────────────────────
        if self.use_industry_embedding:
            ind_emb = self.industry_embedding(industry_ids)  # [B, N, emb_dim]
            stock_features = torch.cat([stock_features, ind_emb], dim=-1)

        # 股票间交互注意力
        interactive_features = self.cross_stock_attention(stock_features)

        # ─── ★ 市场聚合架构 ─────────────────────────
        market_embedding = None
        market_logits = None
        if self.use_market_aggregation:
            # Step 1: 注意力池化 → 市场状态向量
            market_embedding, market_attn = self.market_pooling(
                interactive_features, masks=stock_masks
            )  # [B, market_dim]

            # Step 2: 市场方向预测
            market_logits = self.market_head(market_embedding)  # [B, 1]

            # Step 3: 市场门控调制个股特征
            interactive_flat = interactive_features.view(batch_size * num_stocks, -1)
            modulated_features = self.market_gate(interactive_flat, market_embedding)
            # 重塑
            interactive_flat = modulated_features
        else:
            interactive_flat = interactive_features.view(batch_size * num_stocks, -1)

        # 如果没使用市场聚合，interactive_flat 已被正确设置
        if not self.use_market_aggregation:
            interactive_flat = interactive_features.view(batch_size * num_stocks, -1)

        # 排序特异性变换
        ranking_features = self.ranking_layers(interactive_flat)

        # 生成排序分数
        scores = self.score_head(ranking_features)
        scores = scores.view(batch_size, num_stocks)

        if return_aux:
            aux = {
                'direction': self.direction_head(ranking_features).squeeze(-1),
                'volatility': F.softplus(self.volatility_head(ranking_features)).squeeze(-1),
                'return_abs': self.return_head(ranking_features).squeeze(-1),
            }
            # 市场聚合的输出
            if self.use_market_aggregation:
                aux['market_embedding'] = market_embedding
                aux['market_logits'] = market_logits.squeeze(-1)  # [B]
            return scores, aux

        return scores


# ═══════════════════════════════════════════════════════════════
# ★ 轻量级排序模型 (方案1+2: 统计特征 + GRU + 方向分类)
# ═══════════════════════════════════════════════════════════════

class LightweightStockRanker(nn.Module):
    """
    轻量级股票排序模型 — 专为小样本金融数据设计 (~120K 参数)。

    架构 (25x 参数削减 vs StockTransformer):
      1. 统计矩汇总 (非学习): mean/std/last/trend/min/max → 6F 维固定特征
      2. 共享 GRU (~30K): 1层双向 GRU 捕获残差时序模式
      3. 市场上下文: 简单均值池化 → 拼接
      4. 排序头 (~30K): MLP 输出排序分数
      5. 方向分类头 (~13K): BCE 二分类 (替代回归)
      6. 市场方向头 (~1K): 均值池化 → BCE

    设计原理:
      - 金融时序信噪比极低 (1:13)，深度 Transformer 学到的"模式"大概率是噪声
      - 统计汇总 (动量/波动/趋势) 是量化金融中最鲁棒的信号
      - 25x 参数削减 → 模型只能学到最强信号 → 自然防过拟合
    """

    def __init__(self, input_dim, config, num_stocks=300):
        super(LightweightStockRanker, self).__init__()
        self.model_type = 'LightweightRanker'
        self.config = config
        self.num_stocks = num_stocks

        d_model = config.get('d_model', 128)
        gru_hidden = config.get('gru_hidden', 48)
        dropout = config.get('dropout', 0.2)

        # ─── 行业 Embedding ──────────────────────────
        self.use_industry_embedding = config.get('use_industry_embedding', True)
        if self.use_industry_embedding:
            num_industries = config.get('num_industries', 31)
            industry_emb_dim = config.get('industry_emb_dim', 8)
            self.industry_embedding = nn.Embedding(num_industries, industry_emb_dim)

        # 实际特征维度 (input_dim 是不含 industry 的 scaler 特征数)
        # 模型接收 input_dim + 1 列 (含 industry)，forward 中会去掉 industry
        feat_dim = input_dim

        # ─── 1. 统计矩汇总投影 ──────────────────────
        # 4个统计量 (mean/std/last/trend) × 实际特征维度
        summary_dim = feat_dim * 4
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )

        # ─── 2. 轻量 GRU 时序编码器 ──────────────────
        self.gru_input_proj = nn.Linear(feat_dim, 64)
        self.gru = nn.GRU(
            input_size=64, hidden_size=gru_hidden,
            num_layers=1, batch_first=True,
            bidirectional=True, dropout=0.0
        )
        self.gru_output_proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
        )

        # ─── 3. 合并层 ──────────────────────────────
        combined_dim = d_model + d_model // 2
        self.combine_norm = nn.LayerNorm(combined_dim)
        self.combine_dropout = nn.Dropout(dropout)

        # ─── 4. 市场上下文 (简单均值池化) ────────────
        self.market_context_proj = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.market_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)  # BCE logit: 市场涨/跌
        )

        # ─── 5. 排序头 ──────────────────────────────
        stock_dim = combined_dim + 64  # stock + market context
        self.scorer = nn.Sequential(
            nn.Linear(stock_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )

        # ─── 6. 方向分类头 (方案2: BCE替代回归) ────
        self.direction_head = nn.Sequential(
            nn.Linear(combined_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 2, 1)  # BCE logit
        )

        # ─── 7. 收益回归头 (仅用于门控, 弱训练) ────
        self.return_head = nn.Sequential(
            nn.Linear(combined_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 2, 1)
        )

        # ─── 8. 波动率头 (可选) ─────────────────────
        self.volatility_head = nn.Sequential(
            nn.Linear(combined_dim, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _compute_summaries(self, x):
        """计算时序统计矩 (非学习, 4个统计量)"""
        # x: [B, N, T, nFeat]
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        mean = x.mean(dim=2)           # [B, N, nFeat]
        std = x.std(dim=2).clamp(0, 10)  # [B, N, nFeat]
        last = x[:, :, -1, :]          # [B, N, nFeat]
        first = x[:, :, 0, :]          # [B, N, nFeat]
        trend = last - first            # [B, N, nFeat]

        summaries = torch.cat([mean, std, last, trend], dim=-1)  # [B, N, 4*nFeat]
        return summaries

    def forward(self, src, return_aux=False, stock_masks=None):
        """
        Args:
            src: [B, N, T, F] — batch, stocks, 60 days, features
            return_aux: 返回辅助输出
            stock_masks: [B, N] 有效股票掩码

        Returns:
            scores: [B, N] 排序分数
            aux: (return_aux=True)
        """
        B, N, T, nF = src.shape

        # ─── 提取行业ID ──────────────────────────────
        if self.use_industry_embedding:
            industry_ids = src[:, :, 0, -1].long()  # [B, N]
            src = src[:, :, :, :-1]

        # ─── 1. 统计矩汇总 ──────────────────────────
        summaries = self._compute_summaries(src)  # [B, N, 6*nFeat]
        summary_feat = self.summary_proj(summaries)  # [B, N, d_model]

        # ─── 2. GRU 时序编码 ────────────────────────
        n_feat_val = src.shape[-1]  # 去掉 industry 后的实际特征维度
        src_flat = src.reshape(B * N, T, n_feat_val)
        gru_in = self.gru_input_proj(src_flat)  # [B*N, T, 64]
        _, h_n = self.gru(gru_in)  # h_n: [2, B*N, gru_hidden]
        gru_out = h_n.transpose(0, 1).reshape(B * N, -1)  # [B*N, gru_hidden*2]
        gru_feat = self.gru_output_proj(gru_out)  # [B*N, d_model//2]
        gru_feat = gru_feat.view(B, N, -1)  # [B, N, d_model//2]

        # ─── 3. 合并 ────────────────────────────────
        combined = torch.cat([summary_feat, gru_feat], dim=-1)  # [B, N, combined_dim]
        combined = self.combine_norm(combined)
        combined = self.combine_dropout(combined)

        # ─── 4. 市场上下文 ──────────────────────────
        if stock_masks is not None:
            mask = stock_masks.unsqueeze(-1).float()  # [B, N, 1]
            market_pool = (combined * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            market_pool = combined.mean(dim=1)  # [B, combined_dim]

        market_context = self.market_context_proj(market_pool)  # [B, 64]
        market_logits = self.market_head(market_context)  # [B, 1]

        # 拼接到每个股票
        market_expanded = market_context.unsqueeze(1).expand(B, N, -1)  # [B, N, 64]
        stock_with_market = torch.cat([combined, market_expanded], dim=-1)  # [B, N, combined_dim+64]

        # ─── 5. 排序分数 ────────────────────────────
        stock_flat = stock_with_market.view(B * N, -1)
        scores = self.scorer(stock_flat).view(B, N)  # [B, N]

        if return_aux:
            combined_flat = combined.reshape(B * N, -1)  # [B*N, combined_dim]
            aux = {
                'direction': self.direction_head(combined_flat).squeeze(-1),  # [B*N]
                'return_abs': self.return_head(combined_flat).squeeze(-1),  # [B*N]
                'volatility': F.softplus(self.volatility_head(combined_flat)).squeeze(-1),  # [B*N]
                'market_logits': market_logits.squeeze(-1),  # [B]
            }
            if self.use_industry_embedding:
                aux['industry_emb'] = self.industry_embedding(industry_ids)
            return scores, aux

        return scores
