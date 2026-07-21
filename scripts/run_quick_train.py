"""
Quick training + self-scoring pipeline
- Uses 158+39 features (no fundamentals, which need downloading)
- All other improvements active: data augmentation, new loss, Gumbel NDCG
- Reduced epochs for quick turnaround
"""
import sys, os, json
sys.path.insert(0, 'code/src')

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, 'model', '60_158+39_v2')

# Override config for this run
from config import config
config.update({
    'feature_num': '158+39',
    'num_epochs': 50,
    'early_stopping_patience': 12,
    'output_dir': _MODEL_DIR,
    'seed': 42,
    'batch_size': 4,
    'use_fundamentals': False,  # Skip fundamentals
    'augment_prob': 0.5,
    'time_mask_ratio': 0.15,
    'feature_noise_std': 0.005,
    'stock_dropout_ratio': 0.2,
    'precision_weight': 0.5,
    'use_exact_lambda': True,
    'use_gumbel_ndcg': False,  # Gumbel can be unstable
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 3.0,
    'ndcg_weight': 0.3,
})

os.makedirs(config['output_dir'], exist_ok=True)
with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

print(f"Config saved to {config['output_dir']}/config.json")
print(f"Features: {config['feature_num']}")
print(f"Epochs: {config['num_epochs']}")
print(f"Device: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")

# Now train
from train import main as train_main, feature_cloums_map, feature_engineer_func_map
# Override feature config
config['feature_num'] = '158+39'
# The train.py uses config directly
print("\n=== Starting Training ===")
best_score = train_main()
print(f"\n=== Training Complete! Best final_score: {best_score:.4f} ===")

# Run self-scoring
print("\n=== Running Self-Scoring ===")
from subprocess import run
run([sys.executable, 'score_self.py'], cwd=_PROJECT_ROOT)
