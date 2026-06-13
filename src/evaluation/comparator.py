"""模型对比模块

加载和对比多个模型的性能。
"""

import os
import glob
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib

from src.utils.logger import get_logger
from src.utils.io_utils import load_json, ensure_dir, save_json
from .metrics import MetricsCalculator

logger = get_logger("dry_bean.evaluation.comparator")


class ModelComparator:
    """模型对比工具

    加载多个已训练模型，批量评估并生成对比报告。
    """

    def __init__(self, results_dir: str = './results'):
        """
        Args:
            results_dir: 实验结果目录
        """
        self.results_dir = results_dir
        self.metrics_calculator = MetricsCalculator()

    def compare_all(self, X_test: pd.DataFrame, y_test: pd.Series,
                    class_names: List[str] = None) -> Dict[str, Dict]:
        """对比所有已训练的模型

        Args:
            X_test: 测试集特征
            y_test: 测试集标签
            class_names: 类别名称列表

        Returns:
            模型名称到评估结果的映射
        """
        results = {}

        # 加载所有实验记录
        exp_dir = os.path.join(self.results_dir, 'experiments')
        if not os.path.exists(exp_dir):
            logger.warning(f"实验目录不存在: {exp_dir}")
            return results

        for exp_file in sorted(glob.glob(os.path.join(exp_dir, '*.json'))):
            try:
                exp_data = load_json(exp_file)
                if exp_data.get('status') != 'completed':
                    continue

                model_name = exp_data.get('model_name', 'unknown')

                # 加载模型
                model_path = exp_data.get('artifacts', {}).get('model_path')
                if not model_path:
                    continue

                model = self._load_model(model_name, model_path)
                if model is None:
                    continue

                # 评估
                eval_result = self._evaluate_model(model, X_test, y_test, class_names)
                results[model_name] = eval_result

                logger.info(f"评估 {model_name}: accuracy={eval_result['accuracy']:.4f}")

            except Exception as e:
                logger.error(f"评估实验 {exp_file} 失败: {e}")

        return results

    def _load_model(self, model_name: str, model_path: str):
        """加载模型"""
        try:
            if model_name == 'random_forest':
                from src.models.traditional import RFModel
                return RFModel.load_model(model_path)
            elif model_name == 'xgboost':
                from src.models.traditional import XGBoostModel
                return XGBoostModel.load_model(model_path)
            elif model_name.startswith('pytorch_'):
                from src.models.deep_learning import PyTorchModel
                return PyTorchModel.load_model(model_path)
            elif model_name == 'lightgbm':
                from src.models.advanced import LightGBMModel
                return LightGBMModel.load_model(model_path)
            else:
                logger.warning(f"未知模型类型: {model_name}")
                return None
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {e}")
            return None

    def _evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                       class_names: List[str] = None) -> Dict:
        """评估单个模型"""
        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        metrics = self.metrics_calculator.calculate_all(
            y_test.values, y_pred, class_names=class_names
        )

        # 推理速度测试
        inference_time = self._benchmark_inference(model, X_test)
        metrics['inference_time_ms'] = inference_time

        return metrics

    def _benchmark_inference(self, model, X_test: pd.DataFrame,
                            n_runs: int = 10) -> float:
        """推理速度基准测试"""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X_test[:100] if len(X_test) > 100 else X_test)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转为毫秒

        return float(np.mean(times))

    def generate_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """生成对比表格"""
        rows = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Balanced Accuracy': metrics.get('balanced_accuracy', 0),
                'Precision (Macro)': metrics.get('precision_macro', 0),
                'Recall (Macro)': metrics.get('recall_macro', 0),
                'F1 (Macro)': metrics.get('f1_macro', 0),
                'F1 (Weighted)': metrics.get('f1_weighted', 0),
                'Inference Time (ms)': metrics.get('inference_time_ms', 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values('Accuracy', ascending=False)
