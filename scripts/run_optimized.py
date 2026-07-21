"""
run_optimized.py — 优化流水线：一键执行 Walk-Forward 训练 + 集成预测

这是面向官方评测的完整优化方案入口脚本。

改进汇总:
  P0: Walk-Forward 滚动窗口交叉验证 (6个窗口)
  P1: 分位数 rank 排序标签（消除牛熊市偏差）
  P2: 多周期预测融合（5日平均）
  P3: 轻量化模型（128维/2层/高dropout/强正则化）
  P4: Walk-Forward 自动选择泛化最优配置
  P5: 多模型一致性过滤后处理

用法:
  # 完整流程：训练 + 预测
  python run_optimized.py

  # 仅训练
  python run_optimized.py --train-only

  # 仅预测（需要已有模型）
  python run_optimized.py --predict-only

  # 对比：同时运行 light 和 standard 配置
  python run_optimized.py --configs light,standard

  # 使用多个随机种子
  python run_optimized.py --seeds 42,123,456

输出:
  model/walk_forward/              — 所有窗口的模型
  model/walk_forward/ensemble_result_*.csv  — 最终预测结果
  model/walk_forward/walk_forward_summary.json — 训练汇总
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, 'code', 'src')


def run_walk_forward_train(config='light', seeds='42', data_path=None, output_base=None):
    """运行 Walk-Forward 训练"""
    wf_script = os.path.join(SRC_DIR, 'walk_forward.py')
    cmd = [
        sys.executable, wf_script,
        '--config', config,
        '--seeds', seeds,
    ]
    if data_path:
        cmd.extend(['--data-path', data_path])
    if output_base:
        cmd.extend(['--output', output_base])

    print(f"\n{'='*70}")
    print(f"🔧 执行 Walk-Forward 训练: config={config}, seeds={seeds}")
    print(f"   命令: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def run_ensemble_predict(config='light', seeds='42', multi_period=5,
                          data_path=None, wf_dir=None, output=None):
    """运行集成预测"""
    cmd = [
        sys.executable, os.path.join(PROJECT_DIR, 'walk_forward_predict.py'),
        '--config', config,
        '--seeds', seeds,
        '--multi-period', str(multi_period),
    ]
    if data_path:
        cmd.extend(['--data-dir', data_path])
    if wf_dir:
        cmd.extend(['--wf-dir', wf_dir])
    if output:
        cmd.extend(['--output', output])

    print(f"\n{'='*70}")
    print(f"🎯 执行集成预测: config={config}")
    print(f"   命令: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def parse_float_list(s):
    """解析逗号分隔的数值列表"""
    return [x.strip() for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='沪深300选股优化流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_optimized.py                              # 完整流程(light配置, seed=42)
  python run_optimized.py --train-only                 # 仅训练
  python run_optimized.py --predict-only               # 仅预测
  python run_optimized.py --configs light,standard     # 对比两种配置
  python run_optimized.py --seeds 42,123               # 多seed
  python run_optimized.py --multi-period 3             # 3日预测融合
        """
    )
    parser.add_argument('--configs', type=str, default='light',
                        help='配置名称，逗号分隔 (light, standard, v7)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='随机种子，逗号分隔 (如 42,123,456)')
    parser.add_argument('--multi-period', type=int, default=5,
                        help='多周期预测融合天数 (默认5)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='数据目录 (默认: 项目根目录)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='模型输出目录 (默认: model/walk_forward)')
    parser.add_argument('--train-only', action='store_true',
                        help='仅训练，不预测')
    parser.add_argument('--predict-only', action='store_true',
                        help='仅预测（需要已有模型）')
    parser.add_argument('--final-output', type=str, default=None,
                        help='最终提交文件路径')
    args = parser.parse_args()

    # 路径配置
    data_path = args.data_path or PROJECT_DIR
    output_dir = args.output_dir or os.path.join(PROJECT_DIR, 'model', 'walk_forward')

    configs = parse_float_list(args.configs)
    print(f"\n{'#'*70}")
    print(f"# 沪深300选股优化流水线")
    print(f"# 配置: {configs}  |  Seeds: {args.seeds}  |  多周期: {args.multi_period}")
    print(f"# 数据: {data_path}")
    print(f"# 输出: {output_dir}")
    print(f"{'#'*70}")

    start_time = datetime.now()

    # ─── Phase 1: Training ──────────────────────────────────
    if not args.predict_only:
        print(f"\n{'#'*70}")
        print(f"# Phase 1: Walk-Forward Training")
        print(f"{'#'*70}")

        for config_name in configs:
            success = run_walk_forward_train(
                config=config_name,
                seeds=args.seeds,
                data_path=data_path,
                output_base=output_dir,
            )
            if not success:
                print(f"❌ 配置 {config_name} 训练失败！")
            else:
                print(f"✅ 配置 {config_name} 训练完成！")

    # ─── Phase 2: Prediction ────────────────────────────────
    if not args.train_only:
        print(f"\n{'#'*70}")
        print(f"# Phase 2: Ensemble Prediction")
        print(f"{'#'*70}")

        final_output = args.final_output
        if final_output is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_output = os.path.join(output_dir, f'final_result_{ts}.csv')

        # 为每个配置运行预测
        for config_name in configs:
            config_output = final_output.replace('.csv', f'_{config_name}.csv')
            success = run_ensemble_predict(
                config=config_name,
                seeds=args.seeds,
                multi_period=args.multi_period,
                data_path=data_path,
                wf_dir=output_dir,
                output=config_output,
            )
            if success:
                print(f"✅ {config_name} 预测完成 → {config_output}")
            else:
                print(f"❌ {config_name} 预测失败！")

    elapsed = datetime.now() - start_time
    print(f"\n{'#'*70}")
    print(f"# 流水线完成！总耗时: {elapsed}")
    print(f"{'#'*70}")

    # 打印结果摘要
    summary_path = os.path.join(output_dir, 'walk_forward_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n📊 Walk-Forward 训练汇总:")
        for seed_key, results in summary.get('results', {}).items():
            scores = [r['score'] for r in results]
            if scores:
                print(f"  {seed_key}: 窗口数={len(scores)}, "
                      f"avg_score={sum(scores)/len(scores):.4f}, "
                      f"min={min(scores):.4f}, max={max(scores):.4f}")


if __name__ == '__main__':
    main()
