"""
截断 + 半衰期敏感性实验
Phase 1: W1 截断扫描 (2015, 2018, 2021 vs 2010 baseline)
Phase 2: W1 半衰期扫描 (365, 1825, None vs 730 baseline) on best start date
Phase 3: W6 验证 best config

用法: python run_truncation_exp.py
"""
import subprocess
import sys
import os
import json
from datetime import datetime

PROJECT_DIR = os.environ.get(
    'CSI300_PROJECT_DIR',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PYTHON = os.environ.get('CSI300_PYTHON', sys.executable)
WF_SCRIPT = os.path.join(PROJECT_DIR, 'code', 'src', 'walk_forward.py')
OUTPUT_BASE = os.path.join(PROJECT_DIR, 'model', 'walk_forward_truncation_exp')

# Baseline (already done in walk_forward_v8_2010_full):
# W1 with 2010 data + half-life 730: final_score=0.2068, return=-10.16%

def run_exp(label, data_start=None, half_life=None, windows='1'):
    """Run a single walk_forward experiment."""
    output_dir = os.path.join(OUTPUT_BASE, label)
    cmd = [PYTHON, WF_SCRIPT, '--config', 'v8_improved', '--seeds', '42',
           '--windows', windows, '--output', output_dir]
    if data_start:
        cmd.extend(['--data-start', data_start])
    if half_life is not None:
        cmd.extend(['--half-life', str(half_life)])

    print(f"\n{'='*70}")
    print(f"🔬 {label}: data_start={data_start}, half_life={half_life}")
    print(f"   Output: {output_dir}")
    print(f"{'='*70}")

    start = datetime.now()
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=False)
    elapsed = (datetime.now() - start).total_seconds() / 60

    # Read result
    summary_path = os.path.join(output_dir, 'walk_forward_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        scores = [r['score'] for r in summary['results'].get('seed42', [])]
        mean_score = sum(scores) / len(scores) if scores else 0
    else:
        scores = []
        mean_score = 0

    return {
        'label': label,
        'data_start': data_start,
        'half_life': half_life,
        'scores': scores,
        'mean_score': mean_score,
        'elapsed_min': elapsed,
        'exit_code': result.returncode,
    }


def main():
    results = []

    # ═══════════════════════════════════════════════════
    # Phase 1: W1 截断扫描
    # ═══════════════════════════════════════════════════
    print("\n" + "#"*70)
    print("# Phase 1: W1 截断扫描")
    print("#"*70)

    truncations = [
        ('w1_2015', '2015-01-01', 730),
        ('w1_2018', '2018-01-01', 730),
        ('w1_2021', '2021-01-01', 730),
    ]

    for label, ds, hl in truncations:
        r = run_exp(label, data_start=ds, half_life=hl, windows='1')
        results.append(r)
        print(f"  ✅ {label}: scores={r['scores']}, mean={r['mean_score']:.4f}, "
              f"time={r['elapsed_min']:.0f}min")

    # Find best start date
    best = max(results, key=lambda x: x['mean_score'])
    print(f"\n🏆 Best start date: {best['label']} (score={best['mean_score']:.4f})")

    # ═══════════════════════════════════════════════════
    # Phase 2: W1 半衰期扫描 on best start date
    # ═══════════════════════════════════════════════════
    print("\n" + "#"*70)
    print(f"# Phase 2: W1 半衰期扫描 (data_start={best['data_start']})")
    print("#"*70)

    best_ds = best['data_start']
    half_lives = [
        ('w1_hl365', best_ds, 365),
        ('w1_hl1825', best_ds, 1825),
        ('w1_hlNone', best_ds, None),
    ]

    for label, ds, hl in half_lives:
        r = run_exp(label, data_start=ds, half_life=hl, windows='1')
        results.append(r)
        print(f"  ✅ {label}: scores={r['scores']}, mean={r['mean_score']:.4f}, "
              f"time={r['elapsed_min']:.0f}min")

    # Find best overall config
    best_overall = max(results, key=lambda x: x['mean_score'])
    print(f"\n🏆 Best overall: {best_overall['label']} "
          f"(data_start={best_overall['data_start']}, half_life={best_overall['half_life']}, "
          f"score={best_overall['mean_score']:.4f})")

    # ═══════════════════════════════════════════════════
    # Phase 3: W6 验证
    # ═══════════════════════════════════════════════════
    print("\n" + "#"*70)
    print(f"# Phase 3: W6 验证 best config")
    print("#"*70)

    r = run_exp('w6_best', data_start=best_overall['data_start'],
                half_life=best_overall['half_life'], windows='6')
    results.append(r)
    print(f"  ✅ w6_best: scores={r['scores']}, mean={r['mean_score']:.4f}, "
          f"time={r['elapsed_min']:.0f}min")

    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📊 实验汇总")
    print("="*70)
    print(f"{'Label':<20} {'DataStart':<12} {'HalfLife':<10} {'W1_Score':<10} {'Time(min)':<10}")
    print("-"*70)
    # Add baseline
    print(f"{'BASELINE (2010,hl730)':<20} {'2010':<12} {'730':<10} {'0.2068':<10} {'~60':<10}")
    for r in results:
        score_str = f"{r['mean_score']:.4f}" if r['mean_score'] else "N/A"
        hl_str = str(r['half_life']) if r['half_life'] is not None else 'None'
        print(f"{r['label']:<20} {r['data_start'] or '2010':<12} {hl_str:<10} "
              f"{score_str:<10} {r['elapsed_min']:.0f}")

    # Save
    summary_file = os.path.join(OUTPUT_BASE, 'experiment_summary.json')
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            'baseline': {'data_start': '2010-01-01', 'half_life': 730, 'w1_score': 0.2068},
            'results': results,
            'best': best_overall,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 汇总: {summary_file}")


if __name__ == '__main__':
    main()
