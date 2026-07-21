"""
sync_to_docker.py — 一键同步本地改进到 app/ Docker 提交目录

用法:
    python sync_to_docker.py                  # 同步代码+模型+数据
    python sync_to_docker.py --code-only      # 仅同步代码
    python sync_to_docker.py --model-only     # 仅同步模型
    python sync_to_docker.py --model-dir model/60_158+39+fundamental_v4  # 指定模型目录
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()  # 项目根目录

SRC = ROOT / 'code' / 'src'
APP_SRC = ROOT / 'app' / 'code' / 'src'
APP_MODEL = ROOT / 'app' / 'model' / '60_158+39+fundamental+momentum_v8_improved'
APP_DATA = ROOT / 'app' / 'data'

# 同步的源文件列表（不包括 __pycache__ 和临时文件）
SYNC_FILES = [
    'config.py',
    'config_quick.py',
    'model.py',
    'train.py',
    'utils.py',
    'fundamental.py',
    'ensemble.py',
    'data_loader.py',
    'macro_industry.py',
    'market_gate.py',
    'walk_forward.py',
]


def sync_code():
    """同步代码文件"""
    print('=' * 50)
    print('Syncing code...')
    print('=' * 50)

    for f in SYNC_FILES:
        src = SRC / f
        dst = APP_SRC / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f'  OK: {f} ({src.stat().st_size:,} bytes)')
        else:
            print(f'  SKIP: {f} (not found)')

    # 为 Docker 环境补充 fundamental_path（config.py 使用动态路径，无需路径替换）
    config_file = APP_SRC / 'config.py'
    if config_file.exists():
        content = config_file.read_text(encoding='utf-8')
        if "'fundamental_path'" not in content:
            # 在 data_path 后追加 fundamental_path
            content = content.replace(
                "'data_path': _PROJECT_ROOT,",
                "'data_path': _PROJECT_ROOT,\n    'fundamental_path': os.path.join(_PROJECT_ROOT, 'data', 'fundamentals.csv'),"
            )
            config_file.write_text(content, encoding='utf-8')
            print(f'  OK: config.py — added fundamental_path for Docker')
        else:
            print(f'  OK: config.py — fundamental_path already present')


def sync_model(model_dir=None):
    """同步模型文件"""
    print()
    print('=' * 50)
    print('Syncing model...')
    print('=' * 50)

    if model_dir is None:
        # Auto-discover latest model
        model_root = ROOT / 'model'
        candidates = sorted(
            [d for d in model_root.iterdir() if d.is_dir() and (d / 'best_model.pth').exists()],
            key=lambda d: d.stat().st_mtime, reverse=True
        )
        if not candidates:
            print('  ERROR: No trained model found!')
            return
        model_dir = candidates[0]

    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    print(f'  Source: {model_path}')

    app_model = Path(APP_MODEL)
    app_model.mkdir(parents=True, exist_ok=True)

    for f in ['best_model.pth', 'scaler.pkl', 'config.json']:
        src = model_path / f
        dst = app_model / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f'  OK: {f} ({src.stat().st_size:,} bytes)')
        else:
            print(f'  SKIP: {f} (not found)')

    # Copy final_score.txt if exists
    score_file = model_path / 'final_score.txt'
    if score_file.exists():
        shutil.copy2(score_file, app_model / 'final_score.txt')
        print(f'  OK: final_score.txt')


def sync_data():
    """同步数据文件"""
    print()
    print('=' * 50)
    print('Syncing data...')
    print('=' * 50)

    APP_DATA.mkdir(parents=True, exist_ok=True)

    # Fundamental data
    fund_src = ROOT / 'data' / 'fundamentals.csv'
    fund_dst = APP_DATA / 'fundamentals.csv'
    if fund_src.exists():
        shutil.copy2(fund_src, fund_dst)
        print(f'  OK: fundamentals.csv ({fund_src.stat().st_size:,} bytes)')
    else:
        print(f'  SKIP: fundamentals.csv (not found)')

    # Train/Test data (usually already there, only copy if missing)
    for f in ['train.csv', 'test.csv']:
        dst = APP_DATA / f
        if not dst.exists():
            src = ROOT / f
            if src.exists():
                shutil.copy2(src, dst)
                print(f'  OK: {f} ({src.stat().st_size:,} bytes)')
            else:
                src = ROOT / 'data' / f
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f'  OK: {f} (from data/)')
        else:
            print(f'  SKIP: {f} (already exists)')


def verify():
    """验证同步结果"""
    print()
    print('=' * 50)
    print('Verification')
    print('=' * 50)

    errors = []

    # Check model
    model_pth = APP_MODEL / 'best_model.pth'
    if not model_pth.exists():
        errors.append('best_model.pth missing!')
    else:
        print(f'  Model: {model_pth.stat().st_size:,} bytes')

    # Check code
    for f in ['config.py', 'model.py', 'train.py', 'test.py', 'utils.py', 'fundamental.py']:
        if not (APP_SRC / f).exists():
            errors.append(f'{f} missing!')

    # Check config uses dynamic paths (not hardcoded)
    config_file = APP_SRC / 'config.py'
    if config_file.exists():
        config_content = config_file.read_text(encoding='utf-8')
        if 'C:/Users/' in config_content:
            errors.append('config.py: contains hardcoded Windows paths!')
        if '_PROJECT_ROOT' not in config_content:
            errors.append('config.py: missing _PROJECT_ROOT (dynamic path)')

    # Check data
    for f in ['train.csv', 'test.csv', 'fundamentals.csv']:
        if not (APP_DATA / f).exists():
            errors.append(f'data/{f} missing!')

    if errors:
        print(f'\n  ERRORS ({len(errors)}):')
        for e in errors:
            print(f'    - {e}')
    else:
        py_count = len(list(APP_SRC.glob('*.py')))
        csv_count = len(list(APP_DATA.glob('*.csv')))
        print(f'  Code files: {py_count}')
        print(f'  Data files: {csv_count}')
        print('  All checks passed!')


def main():
    parser = argparse.ArgumentParser(description='Sync local improvements to Docker app/')
    parser.add_argument('--code-only', action='store_true', help='Only sync code')
    parser.add_argument('--model-only', action='store_true', help='Only sync model')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model directory to sync (auto-detect latest if not specified)')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification')
    args = parser.parse_args()

    if args.model_only:
        sync_model(args.model_dir)
    elif args.code_only:
        sync_code()
    else:
        sync_code()
        sync_model(args.model_dir)
        sync_data()

    if not args.no_verify:
        verify()

    print()
    print('Done! Ready for Docker submission.')


if __name__ == '__main__':
    main()
