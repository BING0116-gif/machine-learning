"""评估指标模块

计算各种分类评估指标。
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)

from src.utils.logger import get_logger

logger = get_logger("dry_bean.evaluation")


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray = None,
                     class_names: List[str] = None) -> Dict:
        """计算所有主要指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（可选）
            class_names: 类别名称列表（可选）

        Returns:
            包含所有指标的字典
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }

        # 每类指标
        try:
            metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        except Exception:
            pass

        # 分类报告
        try:
            metrics['classification_report'] = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
        except Exception:
            metrics['classification_report'] = None

        return metrics

    @staticmethod
    def format_metrics(metrics: Dict, decimals: int = 4) -> str:
        """格式化指标为可读字符串"""
        lines = [
            f"Accuracy: {metrics['accuracy']:.{decimals}f}",
            f"Balanced Accuracy: {metrics['balanced_accuracy']:.{decimals}f}",
            f"Precision (Macro): {metrics['precision_macro']:.{decimals}f}",
            f"Recall (Macro): {metrics['recall_macro']:.{decimals}f}",
            f"F1-Score (Macro): {metrics['f1_macro']:.{decimals}f}",
            f"Precision (Weighted): {metrics['precision_weighted']:.{decimals}f}",
            f"Recall (Weighted): {metrics['recall_weighted']:.{decimals}f}",
            f"F1-Score (Weighted): {metrics['f1_weighted']:.{decimals}f}",
        ]
        return '\n'.join(lines)
