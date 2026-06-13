"""随机种子设置模块

确保实验的可复现性，统一设置所有随机数生成器的种子。
"""

import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子以确保可复现性

    设置 Python random、NumPy 和 PyTorch（如果可用）的随机种子。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确定性模式（可能影响性能）
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
