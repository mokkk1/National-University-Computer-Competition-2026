"""
conftest.py — 共享 fixtures 和配置
"""
import sys
import os
import pytest

# 确保 code/src 在 path 中
_SRC = os.path.join(os.path.dirname(__file__), '..', 'code', 'src')
sys.path.insert(0, _SRC)


@pytest.fixture(autouse=True)
def set_torch_threads():
    """限制 PyTorch 线程数加速测试（首次调用后跳过）"""
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # 已经设置过，忽略
